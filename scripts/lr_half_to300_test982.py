"""Isolated final K1 evaluation of the validation-selected checkpoint."""
from pathlib import Path
import os,sys,json,subprocess,hashlib,datetime
R=Path(__file__).resolve().parents[1];B=R/'benchmarks/subj01-v1';P=R/'reports/lr-half-to300';O=P/'test982';PY=R/'runtime/env/bin/python'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 sel=json.loads((P/'selection.json').read_text());assert sel['selected_epoch']==300
 assert sha(Path(sel['weight']))==sel['sha256']
 ids=B/'test_full_ids.txt';assert sha(ids)=='4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441'
 assert not O.exists(),'Existing output: inspect before explicit recovery'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 O.mkdir();env=os.environ.copy();env.update(PYTHONPATH=str(B),CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 run=R/'runs/subject01-author-fp32-seed42-lr5e-5-from100';c=json.loads((R/'configs/stage1.json').read_text())
 cmd=[str(PY),str(B/'neuroadapter_k1.py'),'--no_selection','--split','test','--image_ids',str(ids),'--decoded_stimuli_dir',str(O/'decoded'),'--model_weights_dir',str(run),'--saved_epochs','300','--num_predictions','1','--noise_factor','4','--precision','fp32','--batch_size','1','--denoising_steps','50','--seed','42','--num_workers','0','--export_path',sel['weight']]
 for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:cmd+=['--'+k,str(c[k])]
 (P/'test982-started.json').write_text(json.dumps({'command':cmd,'selected_weight_sha256':sel['sha256'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid()},indent=2))
 with (R/'logs/lr-half-to300-test982-decode.log').open('x') as f:rc=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 (R/'logs/lr-half-to300-test982-decode.exit').write_text(str(rc));assert rc==0
 import numpy as np
 manifest={}
 for i in map(int,ids.read_text().split()):
  p=O/'decoded'/f'image_{i:06d}.npz'
  with np.load(p) as d:assert int(d['image_id'])==i and d['candidate_images'].shape==(1,512,512,3) and np.isfinite(d['candidate_images']).all()
  manifest[p.name]=sha(p)
 assert len(manifest)==982
 (O/'generation-audit.json').write_text(json.dumps(manifest,indent=2))
 # Reuse the frozen K1 evaluator verbatim apart from isolated input/output paths.
 source=(B/'evaluate_k1.py').read_text()
 replacements={"B/'outputs/neuroadapter'":repr(str(O/'decoded')),"out=B/'metrics'/a.method/(a.output_tag or a.ids.stem)":"out=Path("+repr(str(O/'metrics'))+")"}
 # The source expression joins a Path: preserve Path type.
 replacements["B/'outputs/neuroadapter'"]='Path('+repr(str(O/'decoded'))+')'
 for old,new in replacements.items():assert source.count(old)==1;source=source.replace(old,new)
 (O/'evaluator-adaptation.json').write_text(json.dumps({'source_sha256':sha(B/'evaluate_k1.py'),'path_only_replacements':replacements},indent=2))
 wrapper="exec(compile("+repr(source)+","+repr(str(B/'evaluate_k1.py'))+",'exec'),{'__name__':'__main__','__file__':"+repr(str(B/'evaluate_k1.py'))+"})"
 (O/'evaluate-isolated.py').write_text(wrapper)

 with (R/'logs/lr-half-to300-test982-score.log').open('x') as f:rc=subprocess.run([str(PY),str(O/'evaluate-isolated.py'),'--method','neuroadapter','--ids',str(ids),'--feature-batch','16'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 (R/'logs/lr-half-to300-test982-score.exit').write_text(str(rc));assert rc==0
 (P/'test982-scoring-complete.json').write_text(json.dumps({'generation_exit':0,'score_exit':0,'n':982,'atlas_review_delivery_pending':True},indent=2))
if __name__=='__main__':main()
