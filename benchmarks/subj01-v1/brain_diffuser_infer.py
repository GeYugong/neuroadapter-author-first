"""Pinned author Brain-Diffuser VD pipeline, explicit I/O only; check first, generate explicitly."""
import argparse,ast,datetime,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,torch,PIL
from PIL import Image
import torchvision.transforms as tvtrans
B=Path(__file__).resolve().parent;U=B/'upstream/brain-diffuser';O=B/'outputs/brain-diffuser-full982';O.mkdir(parents=True,exist_ok=True)
SRC=U/'scripts/versatilediffusion_reconstruct_images.py';AS=B/'assets/p1-models/brain-diffuser/versatile_diffusion'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def savej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--generate',action='store_true');a=ap.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
if a.generate:
 for kind in ['vdvae','clipvision','cliptext']:assert (B/'processed/brain-diffuser-regression'/kind/'COMPLETE.json').exists(),kind
 assert (B/'outputs/brain-diffuser-vdvae-lowlevel/COMPLETE.json').exists()
 assert not (O/'COMPLETE.json').exists()
os.chdir(U);sys.path.insert(0,str(U/'versatile_diffusion'))
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
cfg=model_cfg_bank()('vd_noema')
# Asset paths only; preserve complete architecture including Optimus and text UNet.
def paths(x):
 if isinstance(x,dict):
  for k,v in list(x.items()):
   if isinstance(v,str) and v.startswith('versatile_diffusion/'):
    x[k]=str(AS/Path(v).name) if '/pretrained/' in v else str(U/v)
   else:paths(v)
 elif isinstance(x,list):
  for v in x:paths(v)
paths(cfg);cfg.args.clip_cfg.args.version=str(B/'assets/clip-vit-large-patch14')
print('CONSTRUCT_COMPLETE_AUTHOR_MODEL',flush=True)
net=get_model()(cfg)
pth=AS/'vd-four-flow-v1-0-fp16-deprecated.pth';sd=torch.load(pth,map_location='cpu')
keys=net.load_state_dict(sd,strict=False)
load={'missing':list(keys.missing_keys),'unexpected':list(keys.unexpected_keys),'checkpoint_tensor_count':len(sd),'model_tensor_count':len(net.state_dict())}
# Original author uses strict=False; enforce no unaccounted omission rather than guessing.
assert not load['missing'] and not load['unexpected'],load
assert all(torch.equal(v.cpu(),sd[k].to(dtype=v.dtype)) for k,v in net.state_dict().items());del sd
net.clip.cuda(0);net.autokl.cuda(0);sampler=DDIMSampler_VD(net);net.autokl.half()
net.model.diffusion_model.device='cuda:1';net.model.diffusion_model.half().cuda(1)
audit={'utc':now(),'pid':os.getpid(),'source_sha256':sha(SRC),'full_checkpoint_sha256':sha(pth),'state_dict_audit':load,'all_state_tensors_equal':True,'full_architecture_retained':True,'devices':{'clip':'cuda:0','autokl':'cuda:0','diffusion_model':'cuda:1','optimus':'cpu'},'protocol':{'steps':50,'cfg':7.5,'strength':.75,'mixing':.4,'K':1,'seed':0,'rng':'torch.manual_seed(0) once before full author row order; no T100 reorder'},'asset_path_only_changes':True,'test_gt_input':False,'software':'p1-author310 torch2.0.1, compatible environment differs from author torch1.12.1'}
savej(B/'bd-vd-load-audit.json',audit);print('LOAD_VERIFIED',flush=True)
if not a.generate:sys.exit(0)
ids=np.load(B/'processed/regression-brain/test_unique_ids.npy');assert len(ids)==982
pred_text=torch.tensor(np.load(B/'processed/brain-diffuser-regression/cliptext/nsd_cliptext_predtest_nsdgeneral.npy')).half().cuda(1)
pred_vision=torch.tensor(np.load(B/'processed/brain-diffuser-regression/clipvision/nsd_clipvision_predtest_nsdgeneral.npy')).half().cuda(1)
assert len(pred_text)==len(pred_vision)==982
ns=dict(PIL=PIL,Image=Image,np=np,torch=torch,tvtrans=tvtrans,net=net,sampler=sampler,pred_text=pred_text,pred_vision=pred_vision,n_samples=1,ddim_steps=50,ddim_eta=0,scale=7.5,xtype='image',ctype='prompt',strength=.75,mixing=.4,sub=1,lowlevel=str(B/'outputs/brain-diffuser-vdvae-lowlevel'),out=str(O),ids=ids)
body=ast.parse(SRC.read_text()).body
exec(compile(ast.Module(body=[n for n in body if isinstance(n,ast.FunctionDef) and n.name=='regularize_image'],type_ignores=[]),str(SRC),'exec'),ns)
loop=next(n for n in body if isinstance(n,ast.For) and ast.unparse(n.target)=='im_id')
class IO(ast.NodeTransformer):
 def visit_Call(self,n):
  self.generic_visit(n)
  if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='Image' and n.func.attr=='open':n.args=[ast.parse("lowlevel+'/'+str(im_id)+'.png'",mode='eval').body]
  if isinstance(n.func,ast.Attribute) and n.func.attr=='save':n.args=[ast.parse("out+'/'+str(int(ids[im_id]))+'.png'",mode='eval').body]
  return n
nodes=IO().visit(ast.Module(body=loop.body,type_ignores=[]));ast.fix_missing_locations(nodes);code=compile(nodes,str(SRC),'exec')
savej(O/'STARTED.json',{**audit,'rows':[{'row':i,'nsd_id0':int(v)} for i,v in enumerate(ids)]})
torch.manual_seed(0);t0=time.time()
for im_id in range(len(ids)):
 marker=O/f'row-{im_id:04d}.pt'
 if marker.exists():
  state=torch.load(marker,map_location='cpu');torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda']);continue
 ns['im_id']=im_id
 with torch.no_grad():exec(code,ns)
 temp=marker.with_suffix('.tmp');torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},temp);os.replace(temp,marker)
 print(json.dumps({'row':im_id,'nsd_id0':int(ids[im_id]),'utc':now(),'elapsed_s':time.time()-t0}),flush=True)
assert len(list(O.glob('*.png')))==982
savej(O/'COMPLETE.json',{**audit,'complete_utc':now(),'exit_status':0,'elapsed_seconds':time.time()-t0,'png_sha256':{p.name:sha(p) for p in O.glob('*.png')}})
