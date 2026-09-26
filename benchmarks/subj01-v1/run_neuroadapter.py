from pathlib import Path
import argparse,json,os,subprocess,sys,numpy as np
p=argparse.ArgumentParser();p.add_argument('--ids',type=Path,required=True);a=p.parse_args()
B=Path(__file__).resolve().parent;R=B.parents[1];out=B/'outputs/neuroadapter';out.mkdir(parents=True,exist_ok=True)
ids=np.loadtxt(a.ids,dtype=int).reshape(-1);missing=[]
for i in ids:
 f=out/f'image_{i:06d}.npz'
 if f.exists():
  with np.load(f) as d:assert int(d['image_id'])==i and d['candidate_images'].shape==(1,512,512,3)
 else:missing.append(int(i))
if not missing:print('All IDs already present');sys.exit(0)
subset=B/'assets/neuroadapter-missing.txt';subset.write_text(''.join(str(i)+'\n' for i in missing))
c=json.loads((R/'configs/stage1.json').read_text());run=R/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
cmd=[sys.executable,str(B/'neuroadapter_k1.py'),'--no_selection','--split','test','--image_ids',str(subset),'--decoded_stimuli_dir',str(out),'--model_weights_dir',str(run),'--saved_epochs','120','--num_predictions','1','--noise_factor','4','--precision','fp32','--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0','--export_path',str(run/'exports/epoch-120/inference.pt')]
for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:cmd+=['--'+k,str(c[k])]
env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
print(json.dumps({'command':cmd,'missing_n':len(missing)}),flush=True)
subprocess.run(cmd,cwd=R,env=env,check=True)
