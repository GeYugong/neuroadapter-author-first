from pathlib import Path
import argparse,os,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--method',choices=['neuroadapter','mindeye1','brain-diffuser','takagi','mindbridge'],required=True);p.add_argument('--ids',type=Path,required=True);p.add_argument('--feature-batch',type=int,default=0);p.add_argument('--output-tag',default=None);a=p.parse_args()
B=Path(__file__).resolve().parent;R=B.parents[1];sys.path.insert(0,str(R))
os.environ['TORCH_HOME']=str(R/'runtime/torch');os.environ['AUTHOR_SWAV_SOURCE']=str(R/'runtime/swav');os.environ['AUTHOR_CLIP_PATH']=str(R.parent/'neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt')
import numpy as np,torch
from PIL import Image
import metric_brain_adapter as m
if a.feature_batch:
 from metric_feature_batches import install
 install(m,a.feature_batch)
assert hashlib.sha256((R/'metric_brain_adapter.py').read_bytes()).hexdigest()==json.loads((B/'startup.json').read_text())['metric_sha256']
ids=np.loadtxt(a.ids,dtype=int).reshape(-1)
gt=torch.stack([torch.from_numpy(np.array(Image.open(B/'assets/gt'/f'{i}.png'))).permute(2,0,1).float()/255 for i in ids])
imgs=[]
for i in ids:
 if a.method=='mindbridge':
  record=json.loads((B/'outputs/mindbridge-records'/f'{i}.json').read_text())
  tensorfile=B/'outputs/mindbridge'/f'{i}.pt'
  assert record['nsd_id0']==int(i) and record['K']==1 and record['selection']=='none'
  assert hashlib.sha256(tensorfile.read_bytes()).hexdigest()==record['native_tensor_sha256']
  native=torch.load(tensorfile,map_location='cpu')
  assert native.shape==(1,1,3,512,512) and native.dtype==torch.float32 and torch.isfinite(native).all()
  assert native.min()>=0 and native.max()<=1
  imgs.append(native[0,0]);continue
 if a.method=='neuroadapter':
  with np.load(B/'outputs/neuroadapter'/f'image_{i:06d}.npz') as d:
   assert int(d['image_id'])==i and d['candidate_images'].shape[0]==1
   img=d['candidate_images'][0].copy()
 else:
  image_dir={'mindeye1':'mindeye1','brain-diffuser':'brain-diffuser-full982','takagi':'takagi','mindbridge':'mindbridge'}[a.method]
  img=np.array(Image.open(B/'outputs'/image_dir/f'{i}.png').convert('RGB'))
 imgs.append(torch.from_numpy(img).permute(2,0,1).float()/255)
pred=torch.stack(imgs);pred=m.resize_images(pred,gt.shape[2:])
out=B/'metrics'/a.method/(a.output_tag or a.ids.stem);out.mkdir(parents=True,exist_ok=True)
with torch.no_grad(): detailed,summary=m.MetricEvaluator(device='cuda').compute_all_metrics(gt,pred)
detailed.insert(0,'nsd_id0',ids);detailed.to_csv(out/'per-image.csv',index=False)
result={'feature_batch':a.feature_batch,'negative_pool':'all requested IDs; unchanged original formula','method':a.method,'n_eval':len(ids),'K':1,'selection':'none','eval_ids_hash':hashlib.sha256(a.ids.read_bytes()).hexdigest(),'mean':summary['mean'].to_dict(),'metric_sha256':hashlib.sha256((R/'metric_brain_adapter.py').read_bytes()).hexdigest(),'GT':'original NSD425x425 RGB uint8/255','prediction_serialization':('Original FP32 tensor; PNG is atlas-only' if a.method=='mindbridge' else 'Existing method-native exported uint8 outputs'),'prediction_resize':'author resize_images bilinear align_corners=False to425 before metric-specific preprocessing'}
(out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
