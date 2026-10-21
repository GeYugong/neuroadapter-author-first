"""Fixed paired retrieval scoring and common image features."""
import hashlib,json,os
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
O=R/'reports/retrieval-evaluation-20261022'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def digest(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(a.tobytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 assert p.resolve().is_relative_to(O.resolve()) and not p.exists(),p
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def normalize(a):
 a=np.asarray(a,dtype=np.float64).reshape(len(a),-1)
 assert np.isfinite(a).all()
 norm=np.linalg.norm(a,axis=1,keepdims=True);assert np.all(norm>0)
 return a/norm
def trials(sim,draws):
 assert sim.shape==(982,982) and draws.shape==(30,300)
 assert np.isfinite(sim).all()
 result=[]
 for idx in draws:
  sub=sim[np.ix_(idx,idx)];labels=np.arange(300)
  row=np.argmax(sub,axis=1);col=np.argmax(sub,axis=0)
  result.append(dict(row_query_top1_percent=float(np.mean(row==labels)*100),
   column_query_top1_percent=float(np.mean(col==labels)*100),
   row_max_ties=int(np.sum(np.sum(sub==sub.max(1,keepdims=True),axis=1)>1)),
   column_max_ties=int(np.sum(np.sum(sub==sub.max(0,keepdims=True),axis=0)>1))))
 return result
def evaluate(a,b,draws,folder,metadata):
 folder=Path(folder);assert not folder.exists();folder.mkdir(parents=True)
 af=np.asarray(a).reshape(len(a),-1);bf=np.asarray(b).reshape(len(b),-1)
 sim=normalize(af)@normalize(bf).T
 rounds=trials(sim,draws)
 values=np.array([[r['row_query_top1_percent'],r['column_query_top1_percent']] for r in rounds])
 audit={'a_shape':list(a.shape),'b_shape':list(b.shape),'a_dtype':str(a.dtype),'b_dtype':str(b.dtype),
  'a_sha256':digest(a),'b_sha256':digest(b),'cosine_sha256':digest(sim),
  'finite':True,'orientation':'rows=query a; columns=query b; matching IDs on diagonal',
  'scalar_dot_max_error':max(abs(float(sim[i,j])-float(np.dot(normalize(af[[i]])[0],normalize(bf[[j]])[0]))) for i,j in [(0,0),(0,1),(1,0),(981,981)])}
 assert audit['scalar_dot_max_error']<1e-10
 np.save(folder/'cosine.npy',sim)
 d=dict(metadata,rows=len(a),pool_size=300,rounds=30,chance_percent=100/300,
  mean_percent=values.mean(0).tolist(),draw_sd_percent=values.std(0,ddof=1).tolist(),
  tie_rule='first maximum in the shared sampled ID order',confidence_interval=None,
  uncertainty='SD across overlapping sampled pools; not independent experimental replication',
  per_round=rounds,audit=audit)
 write(folder/'result.json',d)
 return d
def arithmetic_mean(results,folder,metadata):
 x=np.array([[[t['row_query_top1_percent'],t['column_query_top1_percent']] for t in r['per_round']] for r in results])
 y=x.mean(0)
 write(Path(folder)/'result.json',dict(metadata,rows=982,pool_size=300,rounds=30,
  mean_percent=y.mean(0).tolist(),draw_sd_percent=y.std(0,ddof=1).tolist(),
  per_round_percent=y.tolist(),aggregation='Mean candidate accuracies AFTER independent full-pool top1; no feature or image averaging',candidate_count=len(results)))
def sanity():
 angles_a=np.deg2rad([0,10,90]);angles_b=np.deg2rad([0,45,90])
 a=np.stack([np.cos(angles_a),np.sin(angles_a)],1);b=np.stack([np.cos(angles_b),np.sin(angles_b)],1)
 s=a@b.T
 assert np.argmax(s,axis=1).tolist()==[0,0,2]
 assert np.argmax(s,axis=0).tolist()==[0,1,2]
 rg=np.random.RandomState(123);q=normalize(rg.normal(size=(300,64)));sim=q@q.T
 assert np.array_equal(sim.argmax(1),np.arange(300))
 assert np.array_equal(sim.argmax(0),np.arange(300))
 order=rg.permutation(300)
 assert np.array_equal(sim[np.ix_(order,order)].argmax(1),np.arange(300))
 return dict(asymmetric_direction_test=True,identity=True,joint_permutation=True,
  deliberate_shift_top1_percent=float(np.mean(np.argmax(sim[:,np.roll(np.arange(300),1)],1)==np.arange(300))*100))
class CommonClip:
 def __init__(self,path,device='cuda',batch=16):
  import torch,clip
  from torchvision import transforms as T
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False
  self.torch=torch;self.device=device;self.batch=batch
  self.model,_=clip.load(str(path),device='cpu',jit=False)
  self.model=self.model.float().to(device).eval().requires_grad_(False)
  self.transform=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BILINEAR,antialias=True),
    T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 def __call__(self,images):
  t=self.torch
  if images.ndim==3:images=images[None]
  images=images.float()
  assert images.ndim==4 and images.shape[1]==3 and t.isfinite(images).all()
  assert images.min()>=0 and images.max()<=1
  if images.shape[-2:]!=(425,425):
   images=t.nn.functional.interpolate(images,size=(425,425),mode='bilinear',align_corners=False)
  outputs=[]
  with t.inference_mode():
   for part in images.split(self.batch):
    outputs.append(self.model.encode_image(self.transform(part).to(self.device)).float().cpu().numpy())
  a=np.concatenate(outputs);assert np.isfinite(a).all();return a
