"""Author VDVAE reconstruction with no-content shape/RNG warm-up and explicit I/O."""
import ast,hashlib,json,os,time,datetime,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
B=Path(__file__).resolve().parent;U=B/'upstream/brain-diffuser';O=B/'outputs/brain-diffuser-vdvae-lowlevel';O.mkdir(parents=True,exist_ok=True)
SRC=U/'scripts/vdvae_reconstruct_images.py';W=B/'assets/p1-models/brain-diffuser/vdvae/imagenet64-iter-1600000-model-ema.th';P=B/'processed/brain-diffuser-regression/vdvae/nsd_vdvae_nsdgeneral_pred_sub1_31l_alpha50k.npy'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def savej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
assert not (O/'COMPLETE.json').exists()
sys.path.insert(0,str(U/'vdvae'))
from model_utils import set_up_data,load_vaes
body=ast.parse(SRC.read_text()).body
H=ast.literal_eval(next(n.value for n in body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='H'))
class Dot(dict):
 __getattr__=dict.get
 __setattr__=dict.__setitem__
H=Dot(H);H.restore_ema_path=str(W)
np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
H,preprocess_fn=set_up_data(H);ema_vae=load_vaes(H)
weights=torch.load(W,map_location='cpu');weights={k.removeprefix('module.'):v for k,v in weights.items()};assert all(torch.equal(v.cpu(),weights[k]) for k,v in ema_vae.state_dict().items());del weights
ids=np.load(B/'processed/regression-brain/test_unique_ids.npy');assert len(ids)==982
cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();checks=[]
with torch.no_grad():
 for n in [30,22]:
  ends=[]
  for value in [0.,1.]:
   torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
   data,_=preprocess_fn(torch.full((n,64,64,3),value));px,st=ema_vae.decoder.forward(ema_vae.encoder.forward(data),get_latents=True)
   ends.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
  equal=torch.equal(ends[0][0],ends[1][0]) and all(torch.equal(a,b) for a,b in zip(ends[0][1],ends[1][1]));assert equal
  checks.append({'batch':n,'zero_vs_one_rng_end_exact':equal})
torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
class NoContent(Dataset):
 def __len__(self):return len(ids)
 def __getitem__(self,i):return torch.zeros((64,64,3),dtype=torch.float32)
t0=time.time()
with torch.no_grad():
 for x in DataLoader(NoContent(),30,shuffle=False):
  data,_=preprocess_fn(x);px,stats=ema_vae.decoder.forward(ema_vae.encoder.forward(data),get_latents=True)
ref_latent=stats
ns={'np':np,'torch':torch,'ema_vae':ema_vae,'Image':Image,'batch_size':30,'test_images':NoContent(),'output_dir':str(O)}
funcs=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in ['latent_transformation','sample_from_hier_latents']]
exec(compile(ast.Module(body=funcs,type_ignores=[]),str(SRC),'exec'),ns)
ns['input_latent']=ns['latent_transformation'](np.load(P),ref_latent)
loop=[n for n in body if isinstance(n,ast.For)][-1]
class SavePath(ast.NodeTransformer):
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='im' and n.func.attr=='save':
   n.args=[ast.parse("output_dir + '/' + str(i*batch_size+j) + '.png'",mode='eval').body]
  return n
nodes=SavePath().visit(ast.Module(body=loop.body,type_ignores=[]));ast.fix_missing_locations(nodes);code=compile(nodes,str(SRC),'exec')
audit={'pid':os.getpid(),'start_utc':now(),'source_sha256':sha(SRC),'weights_sha256':sha(W),'prediction_sha256':sha(P),'strict_model_tensors_equal':True,'batch_size':30,'seed':0,'rng_checks':checks,'gt_read':False,'shape_source':'all-zero input through original encoder/decoder at original 982/30 batch schedule; content never used','rng_disclosure':'Original script encodes all GT only for final stats shapes and RNG consumption. Here equal-sized zero inputs preserve random draw schedule; constant0/1 RNG endpoint checks pass for batches30/22. Explicit seed0 is engineering disclosure: original script has H.seed0 but never seeds via set_up_hyperparams. Not claiming historical pixel equality.','kind':'VDVAE low-level intermediary, not final Brain-Diffuser','id_order':'original author first-occurrence test982','rows':[{'row':i,'nsd_id0':int(x),'png':f'{i}.png'} for i,x in enumerate(ids)]}
savej(O/'STARTED.json',audit)
for i in range(int(np.ceil(len(ids)/30))):
 marker=O/f'batch-{i:04d}.pt'
 if marker.exists():
  state=torch.load(marker,map_location='cpu');torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda']);continue
 ns['i']=i
 with torch.no_grad():exec(code,ns)
 torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},marker)
 print(json.dumps({'batch':i,'completed':min((i+1)*30,len(ids)),'utc':now()}),flush=True)
assert len(list(O.glob('*.png')))==982
savej(O/'COMPLETE.json',{**audit,'complete_utc':now(),'elapsed_seconds':time.time()-t0,'exit_status':0,'png_sha256':{p.name:sha(p) for p in sorted(O.glob('*.png'))}})
print('LOWLEVEL_COMPLETE',flush=True)
