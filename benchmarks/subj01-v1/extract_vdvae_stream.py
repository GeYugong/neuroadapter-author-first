"""Thin I/O adapter: execute pinned author VDVAE preprocessing and latent body unchanged."""
import ast,datetime,hashlib,json,os,time,sys
from pathlib import Path
import h5py,numpy as np,torch
B=Path(__file__).resolve().parent; U=B/'upstream/brain-diffuser'; O=B/'processed/brain-diffuser-features';O.mkdir(parents=True,exist_ok=True)
SRC=U/'scripts/vdvae_extract_features.py'; W=B/'assets/p1-models/brain-diffuser/vdvae/imagenet64-iter-1600000-model-ema.th'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def savej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
assert not (O/'VDVAE_COMPLETE.json').exists(),'Already complete; do not repeat'
sys.path.insert(0,str(U/'vdvae'))
from model_utils import set_up_data,load_vaes
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
source=SRC.read_text(); tree=ast.parse(source)
hnode=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='H' for t in n.targets))
H=ast.literal_eval(hnode.value)
class Dot(dict):
 __getattr__=dict.get
 __setattr__=dict.__setitem__
H=Dot(H);H.restore_ema_path=str(W)
# Author H.seed exists but original extraction never calls set_up_hyperparams.
# Explicit operational seed for reproducible extraction, disclosed; not claiming historical RNG identity.
np.random.seed(H.seed);torch.manual_seed(H.seed);torch.cuda.manual_seed_all(H.seed)
H,preprocess_fn=set_up_data(H);ema_vae=load_vaes(H)
state=torch.load(W,map_location='cpu');state={k.removeprefix('module.'):v for k,v in state.items()}
assert set(state)==set(ema_vae.state_dict())
assert all(torch.equal(v.cpu(),state[k]) for k,v in ema_vae.state_dict().items());del state
loop=next(n for n in tree.body if isinstance(n,ast.For) and 'testloader' in ast.unparse(n.iter))
body=next(n for n in loop.body if isinstance(n,ast.With))
compiled=compile(ast.Module(body=[body],type_ignores=[]),str(SRC),'exec')
ns=dict(torch=torch,np=np,ema_vae=ema_vae,num_latents=31,batch_size=30,i=0)
class Images(Dataset):
 def __init__(self,ids):self.ids=ids;self.f=h5py.File('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/stimuli/nsd_stimuli.hdf5','r')
 def __len__(self):return len(self.ids)
 def __getitem__(self,idx):
  im=self.f['imgBrick'][int(self.ids[idx])].astype(np.uint8)
  img=Image.fromarray(im);img=T.functional.resize(img,(64,64));return torch.tensor(np.array(img)).float()
audit={'pid':os.getpid(),'start_utc':now(),'source':str(SRC),'source_sha256':sha(SRC),'weights_sha256':sha(W),'strict_all_model_tensors_equal':True,'batch_size':30,'num_latents':31,'order':['test','train'],'seed':0,'seed_disclosure':'Original extraction H.seed=0 is unused (no set_up_hyperparams call); this adapter explicitly seeds NumPy/Torch/CUDA before model construction. No claim of identical historical stochastic feature samples.','implementation':'Author H dict, set_up_data/load_vaes and original AST no_grad latent extraction body; only streamed image I/O, checkpoint/resume and output storage adapted.','device':'physical GPU0','command':'CUDA_VISIBLE_DEVICES=0 envs/p1-author310/bin/python extract_vdvae_stream.py'}
savej(O/'VDVAE_STARTED.json',audit)
t0=time.time();total=0
for split in ['test','train']:
 ids=np.load(B/'processed/regression-brain'/f'{split}_unique_ids.npy');d=O/f'vdvae-{split}-batches';d.mkdir(exist_ok=True)
 loader=DataLoader(Images(ids),30,shuffle=False)
 iterator=iter(loader);existing=sorted(d.glob('batch-*.pt'))
 if existing:
  last=torch.load(existing[-1],map_location='cpu');torch.set_rng_state(last['rng_cpu']);torch.cuda.set_rng_state_all(last['rng_cuda']);np.random.set_state(last['rng_numpy'])
 for j,x in enumerate(iterator):
  p=d/f'batch-{j:04d}.pt';batchids=ids[j*30:(j+1)*30]
  if p.exists():
   old=torch.load(p,map_location='cpu');assert np.array_equal(old['ids'],batchids);continue
  data_input,target=preprocess_fn(x);ns.update(data_input=data_input,test_latents=[])
  exec(compiled,ns);feat=ns['test_latents'][0];assert feat.shape[0]==len(batchids) and np.isfinite(feat).all()
  out={'ids':batchids,'features':feat,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state()}
  temp=p.with_suffix('.tmp');torch.save(out,temp);os.replace(temp,p)
  total+=len(batchids)
  print(json.dumps({'split':split,'batch':j,'shape':list(feat.shape),'dtype':str(feat.dtype),'elapsed_s':time.time()-t0,'utc':now()}),flush=True)
  if j==0:savej(O/f'vdvae-{split}-first-batch.json',{'shape':list(feat.shape),'finite':True,'dtype':str(feat.dtype),'ids':batchids.tolist(),'utc':now()})
 arr=np.concatenate([torch.load(p,map_location='cpu')['features'] for p in sorted(d.glob('batch-*.pt'))]);assert len(arr)==len(ids)
 np.save(O/f'vdvae_{split}_latents.npy',arr)
np.savez(O/'nsd_vdvae_features_31l.npz',train_latents=np.load(O/'vdvae_train_latents.npy'),test_latents=np.load(O/'vdvae_test_latents.npy'))
savej(O/'VDVAE_COMPLETE.json',{**audit,'complete_utc':now(),'elapsed_seconds':time.time()-t0,'exit_status':0,'outputs':{p.name:sha(p) for p in O.glob('*.np*')}})
print('VDVAE_COMPLETE',flush=True)
