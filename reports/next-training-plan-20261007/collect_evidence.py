from pathlib import Path
import json, subprocess, hashlib, datetime, csv, statistics, shutil
r=Path('/data1/matengyu/geyugong/neuroadapter-author-first')
files=['reports/training-diagnostics-20261006/REPORT_ZH.md','reports/training-diagnostics-20261006/training_epochs.csv','reports/training-diagnostics-20261006/validation_metrics.csv','reports/lr-half-to500/training-final-audit.json','reports/lr-half-to500/selection.json','reports/brain-correspondence-l120/README.md','reports/lr-half-from100/README.md','train_brain_adapter.py','brain_adapter/model.py','brain_adapter/loss.py','brain_adapter/ip_adapter/attention_processor.py','brain_adapter/ip_adapter/ip_adapter.py','branch_lr_resume.py']
rows=list(csv.DictReader((r/files[2]).open()))
base=next(x for x in rows if x['epoch']=='400')
epochs=list(csv.DictReader((r/files[1]).open()))
seconds=statistics.mean(float(x['seconds']) for x in epochs if 301<=int(x['epoch'])<=500)
run=r/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
ck={}
for epoch in [400,450,500]:
    p=run/f'checkpoint-{epoch}'
    ck[str(epoch)]={'complete':(p/'COMPLETE').exists(),'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file()),'small_files':[f.name for f in p.iterdir() if f.is_file() and f.stat().st_size<1000000]}
a={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'plan_only_not_started','new_training_updates_this_turn':0,'new_model_forward_this_turn':0,'objective':'balanced_semantics_and_structure','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'git_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=r,text=True).strip(),'tracked_status':subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=r,text=True).strip(),'sources':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files},'existing_checkpoint_inventory':ck,'raw_val500_l400':base,'mean_seconds_epochs_301_to_500':seconds,'estimated_training_loop_hours_50_epochs':50*seconds/3600,'estimated_training_loop_hours_100_epochs':100*seconds/3600,'disk_usage':dict(zip(['total','used','free'],shutil.disk_usage(r))),'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),'gpu_compute_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),'paper':'https://arxiv.org/html/2509.23566v2','proposed_lr':1e-5,'initial_new_epochs':50,'conditional_max_new_epochs':100,'updates_per_epoch':532,'balanced_gate':{'all_non_decreasing':['S','PixCorr','SSIM'],'meaningful_gain_any_of':[{'S_pp':0.3},{'PixCorr_absolute':0.005,'SSIM_absolute':0.005}],'identification_max_decline_pp':0.5,'Eff_SwAV_max_increase_absolute':0.005,'statistical_significance_test':False}}
print(json.dumps(a,ensure_ascii=False,indent=2))
