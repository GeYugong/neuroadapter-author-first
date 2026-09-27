"""Takagi CVPR SD1.4 feature extraction: author body unchanged, streamed explicit NSD IDs."""
import argparse,ast,datetime,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np,torch,PIL
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
B=Path(__file__).resolve().parent;U=B/'upstream/takagi';SD=U/'codes/diffusion_sd1/stable-diffusion';SRC=U/'codes/utils/img2feat_sd.py';O=B/'processed/takagi-features';O.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(U/'codes/utils'),str(SD)]
from nsd_access import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
RAW=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def savej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');ap.add_argument('--probe-one',action='store_true');a=ap.parse_args()
train=np.load(B/'processed/regression-brain/train_unique_ids.npy');test=np.load(B/'processed/regression-brain/test_unique_ids.npy');ids=np.sort(np.concatenate([train,test]));assert len(ids)==len(np.unique(ids))==9841
nsda=NSDAccess(str(RAW));nsda.stimuli_file=str(RAW/'stimuli/nsd_stimuli.hdf5');nsda.stimuli_description_file=str(RAW/'experiments/nsd/nsd_stim_info_merged.csv');nsda.coco_annotation_file=str(B/'assets/coco-official/{}_{}.json')
# Native NSDAccess multi-ID lookup returns the exact per-image pycocotools order.
captionfile=O/'captions-native.json'
if captionfile.exists():caps=json.loads(captionfile.read_text())
else:
 allcaps=nsda.read_image_coco_info(ids.tolist(),info_type='captions')
 caps={str(int(i)):v for i,v in zip(ids,allcaps)}
 for i in ids[[0,-1]]:assert nsda.read_image_coco_info([int(i)],info_type='captions')==caps[str(int(i))]
 assert all(len(v)>0 for v in caps.values());savej(captionfile,caps)
assert set(caps)==set(map(str,ids.tolist()))
np.save(O/'extraction_ids.npy',ids);np.save(O/'train_unique_ids.npy',train);np.save(O/'test_unique_ids.npy',test)
plan={'utc':now(),'source_sha256':sha(SRC),'scope':'Author frozen SD1.4 feature-target extraction; GT allowed only here, never reconstruction conditioning','n_ids':9841,'train_unique':8859,'test_unique':982,'order':'Ascending NSD ID0 union of needed train/test IDs; outputs named by ID and regression keeps native order','seed':42,'rng_disclosure':'Author main seeds42 once per contiguous requested range. This adapter seeds42 once for ascending selected9841 IDs, omitting unrelated73k IDs; stochastic posterior/decode history differs from a full contiguous73k execution. No numerical identity with unknown historical range schedule claimed. Original extra sampler.decode is retained even though output unused.','caption_source':'Original NSDAccess/pycocotools annotations trainval2017, no curated captions; native list order unchanged','caption_sha256':sha(captionfile),'resolution':320,'latent_dim':6400,'caption_mean':True,'steps':50,'strength':.8,'scale':5.,'batch':1,'precision':'author autocast inside no_grad/ema scope; original latent encode outside autocast retained','cpu_imports_pass':True,'gpu_started':False}
savej(B/'takagi-feature-plan.json',plan)
if not a.extract:print(json.dumps(plan),flush=True);sys.exit(0)
assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1'];assert not (O/'COMPLETE.json').exists()
# Replace only native caption lookup I/O with preloaded results, validated above.
nsda.read_image_coco_info=lambda image_index,info_type='captions':caps[str(int(image_index[0]))]
seed_everything(42);torch.cuda.set_device(0)
body=ast.parse(SRC.read_text()).body
ns=dict(torch=torch,np=np,PIL=PIL,Image=Image,instantiate_from_config=instantiate_from_config)
exec(compile(ast.Module(body=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in ['load_model_from_config','load_img_from_arr']],type_ignores=[]),str(SRC),'exec'),ns)
config=OmegaConf.load(SD/'configs/stable-diffusion/v1-inference.yaml');config.model.params.cond_stage_config.params={'version':str(B/'assets/clip-vit-large-patch14')}
model=ns['load_model_from_config'](config,str(B/'assets/p1-models/takagi/sd-v1-4.ckpt'),0,verbose=True)
device=torch.device('cuda:0');model=model.to(device);sampler=DDIMSampler(model);sampler.make_schedule(ddim_num_steps=50,ddim_eta=0.,verbose=False)
ns.update(nsda=nsda,repeat=repeat,model=model,sampler=sampler,device=device,resolution=320,batch_size=1,precision_scope=autocast,t_enc=40,scale=5.,output=str(O))
main=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name=='main');loop=next(n for n in main.body if isinstance(n,ast.For))
class Paths(ast.NodeTransformer):
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np' and n.func.attr=='save':
   old=ast.unparse(n.args[0]);folder='init_latent' if 'init_latent' in old else 'c';n.args[0]=ast.parse("output+'/"+folder+"/'+f'{s:06}.npy'",mode='eval').body
  return n
code=Paths().visit(ast.Module(body=loop.body,type_ignores=[]));ast.fix_missing_locations(code);code=compile(code,str(SRC),'exec')
for f in ['init_latent','c','rng']:(O/f).mkdir(exist_ok=True)
savej(O/'STARTED.json',{**plan,'gpu_started':True,'pid':os.getpid(),'utc':now()});t0=time.time()
for j,s in enumerate(ids.tolist()):
 marker=O/'rng'/f'{s:06}.pt'
 if marker.exists():
  state=torch.load(marker,map_location='cpu');random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda']);continue
 ns['s']=s;exec(code,ns)
 assert ns['init_latent'].shape==(6400,) and ns['c'].shape==(59136,);assert np.isfinite(ns['init_latent']).all() and np.isfinite(ns['c']).all()
 tmp=marker.with_suffix('.tmp');torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},tmp);os.replace(tmp,marker)
 print(json.dumps({'row':j,'id':s,'elapsed_s':time.time()-t0}),flush=True)
 if a.probe_one:
  savej(O/'PROBE_ONE.json',{'nsd_id0':s,'init_latent_shape':list(ns['init_latent'].shape),'c_shape':list(ns['c'].shape),'finite':True,'elapsed_s':time.time()-t0,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'utc':now(),'resume':'First image outputs/RNG retained; full run skips this completed ID and restores saved RNG.'});sys.exit(0)
savej(O/'COMPLETE.json',{**plan,'gpu_started':True,'complete_utc':now(),'exit_status':0,'elapsed_s':time.time()-t0,'n_latents':len(list((O/'init_latent').glob('*.npy'))),'n_captions':len(list((O/'c').glob('*.npy')))})
