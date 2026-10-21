"""Native retrieval for frozen MindEye1 and MindBridge brain image-feature heads."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import ast,functools,json,time,sys,traceback,gc,hashlib,io
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import transforms as T
from retrieval_common import R,O,sha,write,evaluate
B=R/'benchmarks/subj01-v1'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.32);os.nice(10)
def compiled_brain():
 p=B/'upstream/mindeye1/src/models.py';tree=ast.parse(p.read_text())
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BrainNetwork')
 ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,partial=functools.partial)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 return ns['BrainNetwork'],p
def load_inputs(subject,method):
 ids=np.load(O/f'ids-s{subject}.npy');audit=[]
 if subject==1:
  z=np.load(B/'assets/subject1_test_brain.npz');lookup={int(i):j for j,i in enumerate(z['nsd_id0'])}
  vox=z['voxels'][[lookup[int(i)] for i in ids]].copy()
  rows=json.loads((B/'mindbridge-test-inputs.json').read_text())['rows'];rows={x['nsd_id0']:x for x in rows}
  images=[];mb=[]
  if method=='mindbridge_s1':
   from mindbridge_native8 import compile_official
   ns=compile_official()
  for i in ids:
   row=rows[int(i)];jp=R/'reports/native-protocols-20261016/mindbridge/selector-jpg'/(row['sample']+'.jpg')
   with Image.open(jp) as im:images.append(torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()/255)
   if method=='mindbridge_s1':
    bp=B/'assets/mindbridge/data/webdataset_avg_split/test/subj01'/(row['sample']+'.nsdgeneral.npy')
    assert sha(bp)==row['brain_sha256'];a=torch.from_numpy(np.load(bp)).float()
    mb.append(ns['pool_voxels'](a,8192,'max').mean(0))
   audit.append(dict(image_id=int(i),native_jpeg_sha256=sha(jp),brain_sha256=row['brain_sha256']))
  x=torch.stack(mb) if method=='mindbridge_s1' else torch.from_numpy(vox)
 else:
  from mindeye_subject02_common import sample
  parent=R/'reports/mindeye-subject02-test-20261020';auditdata=json.loads((parent/'test-input-audit.json').read_text())
  bp=parent/'test-brain.npz';assert sha(bp)==auditdata['brain_sha256']
  z=np.load(bp);lookup={int(i):j for j,i in enumerate(z['ids'])};x=torch.from_numpy(z['voxels'][[lookup[int(i)] for i in ids]].copy())
  rows={int(row['image_id']):row for row in auditdata['rows']};images=[]
  for j,i in enumerate(ids):
   brain,img,actual=sample(rows[int(i)])
   assert actual==int(i) and torch.equal(brain.mean(0).float(),x[j])
   images.append(img);audit.append(dict(image_id=int(i),native_jpeg_sha256=rows[int(i)]['members']['jpg']['sha256'],brain_sha256=rows[int(i)]['members']['nsdgeneral.npy']['sha256']))
 assert x.shape==(982,8192 if method=='mindbridge_s1' else (15724 if subject==1 else 14278))
 return ids,x,torch.stack(images),audit
def main(method):
 subject=2 if method.endswith('s2') else 1;dest=O/'native'/method
 assert not dest.exists()
 ids,x,images,audit=load_inputs(subject,method)
 if method=='mindbridge_s1':
  from mindbridge_native8 import compile_official,sources
  sources();cls=compile_official()['MindSingle']
  model=cls(in_dim=8192,out_dim_image=257*768,out_dim_text=77*768,h=2048,n_blocks=4,subj_list=[1])
  weight=B/'upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42/last.pth';expected='5bd75a992755d70183633df514754b9084b0a4a681284afc83621b2387aac9ad'
  ck=torch.load(weight,map_location='cpu',mmap=True,weights_only=False)
  model.load_state_dict(ck['model_state_dict'],strict=True)
  source=B/'upstream/mindbridge/src/models.py';which=0;unused=[];aa=None
 else:
  cls,source=compiled_brain();model=cls(in_dim=x.shape[1],out_dim=257*768)
  weight=(B/'assets/mindeye/mindeye_models/prior_257_final_subj01_bimixco_softclip_byol/last.pth' if subject==1 else R/'runs/mindeye-subject02-high/last.pth')
  expected='9323bfc19cd39bb4511200f65fc9f193f1239e98df07a8e78f11c6cbdafe14e4' if subject==1 else '6a07146de13096ae8b6db6b35f8dbc254a5e6035bb2261611722cea8ecfc84f6'
  ck=torch.load(weight,map_location='cpu',mmap=True,weights_only=False)
  sd={k.removeprefix('voxel2clip.'):v for k,v in ck['model_state_dict'].items() if k.startswith('voxel2clip.')}
  unused=[]
  if 'temp' in sd:unused=['temp'];del sd['temp']
  model.load_state_dict(sd,strict=True);del sd
  which=1;aa=False
 assert sha(weight)==expected
 del ck;model.eval().requires_grad_(False).cuda();parts=[]
 with torch.inference_mode():
  for start in range(0,982,8):
   a=model(x[start:start+8].cuda())[which].float().reshape(-1,257,768)
   parts.append(a.cpu().numpy())
   if start%160==0:print('BRAIN',method,start,flush=True)
 a=np.concatenate(parts);del model;torch.cuda.empty_cache();gc.collect()
 from transformers import CLIPVisionModelWithProjection
 clip_path=B/'assets/clip-vit-large-patch14'
 encoder=CLIPVisionModelWithProjection.from_pretrained(str(clip_path),local_files_only=True,attn_implementation='eager').float().eval().requires_grad_(False).cuda()
 pre=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC,antialias=aa),T.CenterCrop(224),
  T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 parts=[]
 with torch.inference_mode():
  for start in range(0,982,8):
   value=encoder(pre(images[start:start+8]).cuda()).last_hidden_state
   value=encoder.visual_projection(encoder.vision_model.post_layernorm(value))
   value=value/torch.linalg.norm(value[:,0],dim=-1)[:,None,None]
   parts.append(value.float().cpu().numpy())
   if start%160==0:print('IMAGE',method,start,flush=True)
 b=np.concatenate(parts);del encoder;torch.cuda.empty_cache();gc.collect()
 assert a.shape==b.shape==(982,257,768)
 result=evaluate(a,b,np.load(O/f'draws-s{subject}.npy'),dest,dict(method=method,subject=subject,kind='native',
  direction_names=['Image (brain-to-image)','Brain (image-to-brain)'],
  feature=('MindEye BrainNetwork projector' if method.startswith('mindeye') else 'MindBridge MindSingle image head')+' versus native projected CLIP-L/14 last hidden257x768 image tokens',
  precision='FP32 brain/image forward, TF32 disabled; FP64 cosine; differs from historical FP16 evaluation',
  inputs='Same fixed brain inputs as reconstruction; corresponding original official test JPEG256 images',
  image_preprocess=dict(resize=224,interpolation='bicubic',antialias=aa,center_crop=224,normalization='CLIP',token_norm='divide by CLS token norm'),
  checkpoint_sha256=expected,unused_checkpoint_parameters=unused,no_GT_selection=True,
  source_sha256={str(source):sha(source),str(Path(__file__)):sha(__file__)}))
 with (dest/'inputs.jsonl').open('x') as f:
  for row in audit:f.write(json.dumps(row)+'\n')
 assert sha(weight)==expected
 write(O/f'native-{method}-exit.json',dict(returncode=0,time=time.time(),mean=result['mean_percent']))
 print('NATIVE_COMPLETE',method,result['mean_percent'],flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('method',choices=['mindeye1_s1','mindeye1_s2','mindbridge_s1']);args=p.parse_args()
 try:main(args.method)
 except BaseException:
  write(O/f'native-{args.method}-failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
