"""Run author CLIP regressions unchanged with explicit benchmark I/O."""
from pathlib import Path
import argparse,hashlib,json,os,time
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['vision','text'],required=True);a=p.parse_args()
F=B/'processed/brain-diffuser-features';D=B/'processed/regression-brain';O=B/'processed/brain-diffuser-regression'/('clip'+a.kind);O.mkdir(parents=True,exist_ok=True)
assert (F/'CLIP_COMPLETE.json').exists()
assert not (O/'STARTED.json').exists(),'Inspect existing run; do not overwrite'
source=B/'upstream/brain-diffuser/scripts'/f'clip{a.kind}_regression.py'
code=source.read_text();start=time.time()
replacements={
 "'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)":repr(str(D/'nsd_train_fmriavg_nsdgeneral_sub1.npy')),
 "'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)":repr(str(D/'nsd_test_fmriavg_nsdgeneral_sub1.npy')),
 "'data/extracted_features/subj{:02d}/nsd_clip"+a.kind+"_train.npy'.format(sub)":repr(str(F/f'nsd_clip{a.kind}_train.npy')),
 "'data/extracted_features/subj{:02d}/nsd_clip"+a.kind+"_test.npy'.format(sub)":repr(str(F/f'nsd_clip{a.kind}_test.npy')),
 "'data/predicted_features/subj{:02d}/nsd_clip"+a.kind+"_predtest_nsdgeneral.npy'.format(sub)":repr(str(O/f'nsd_clip{a.kind}_predtest_nsdgeneral.npy')),
 "'data/regression_weights/subj{:02d}/clip"+a.kind+"_regression_weights.pkl'.format(sub)":repr(str(O/f'clip{a.kind}_regression_weights.pkl'))}
for old,new in replacements.items():
 assert code.count(old)==1,old
 code=code.replace(old,new)
record={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'changes':'I/O paths only; original per-token Ridge fit/predict/full982 calibration and diagnostic score unchanged','kind':a.kind,'start_unix':start,'pid':os.getpid(),'threads':os.environ.get('OMP_NUM_THREADS')}
(O/'STARTED.json').write_text(json.dumps(record,indent=2))
import sys
sys.argv=[str(source),'--sub','1'];ns={'__name__':'__main__'}
exec(compile(code,str(source),'exec'),ns)
import numpy as np
assert np.isfinite(ns['pred_clip']).all()
np.savez(O/'brain-normalization.npz',mean=ns['norm_mean_train'],std_ddof1=ns['norm_scale_train'])
record.update(exit_status=0,elapsed_seconds=time.time()-start,prediction_shape=list(ns['pred_clip'].shape))
(O/'COMPLETE.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)