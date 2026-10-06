
import sys,os,json,hashlib,time
from pathlib import Path
r=Path("/data1/matengyu/geyugong/neuroadapter-author-first")
b=r/"benchmarks/subj01-v1"
a=r/"runtime/imagenet-retrieval-local-fixed-v1"
out=r/"runtime/imagenet-retrieval-range-audit-20261006"
old=r.parent/"neuroadapter-subject1-research"
os.environ["TORCH_HOME"]=str(r/"runtime/torch")
os.environ["AUTHOR_SWAV_SOURCE"]=str(r/"runtime/swav")
os.environ["AUTHOR_CLIP_PATH"]=str(old/"models/evaluation/clip/ViT-L-14.pt")
os.environ["OMP_NUM_THREADS"]="4"
sys.path[:0]=[str(r),str(b)]
import numpy as np,torch
from PIL import Image
import metric_brain_adapter as metric
from metric_feature_batches import install
torch.set_num_threads(4)
ids=np.loadtxt(b/"test_full_ids.txt",dtype=np.int64)
pred=torch.stack([torch.from_numpy(np.load(a/"retrieved-view-tensors"/f"{i}.npy")) for i in ids])
gt=torch.stack([torch.from_numpy(np.array(Image.open(b/"assets/gt"/f"{i}.png").convert("RGB"))).permute(2,0,1).float()/255 for i in ids])
audit={"n":len(ids),"old_min":float(pred.min()),"old_max":float(pred.max()),"above_one_pixels":int((pred>1).sum()),"images_above_one":int((pred.amax(dim=(1,2,3))>1).sum()),"gt_min":float(gt.min()),"gt_max":float(gt.max()),"old_identification_would_divide_all_predictions_by_255":bool(pred.max()>1),"correction":"Clamp existing float32 retrieval views to [0,1]; no quantization, no retrieval rerun, no changed query, mask or ensemble","max_abs_correction":float((pred-pred.clamp(0,1)).abs().max()),"metric_sha256":hashlib.sha256((r/"metric_brain_adapter.py").read_bytes()).hexdigest()}
assert audit["max_abs_correction"]<1e-6
(out/"input-audit.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
install(metric,16)
with torch.inference_mode():
 detailed,summary=metric.MetricEvaluator(device="cuda:0").compute_all_metrics(gt,pred.clamp(0,1))
detailed.insert(0,"nsd_id0",ids)
detailed.to_csv(out/"per-image.csv",index=False)
result={"n_eval":len(ids),"mean":summary["mean"].to_dict(),"input_audit":audit,"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
(out/"metrics.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
