"""Author VDVAE Ridge body unchanged; explicit new input/output paths and missing pickle import."""
import ast,datetime,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np,sklearn,sklearn.linear_model as skl
B=Path(__file__).resolve().parent;F=B/'processed/brain-diffuser-features';D=B/'processed/regression-brain';O=B/'processed/brain-diffuser-regression/vdvae';O.mkdir(parents=True,exist_ok=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def savej(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
assert (F/'VDVAE_COMPLETE.json').exists()
assert not (O/'COMPLETE.json').exists(),'Already complete'
assert not (O/'STARTED.json').exists(),'Existing run requires explicit investigation'
source=B/'upstream/brain-diffuser/scripts/vdvae_regression.py'
start={'utc':now(),'pid':os.getpid(),'source_sha256':sha(source),'sklearn':sklearn.__version__,'alpha':50000,'max_iter':10000,'fit_intercept':True,'solver':'author default auto','cpu_threads':os.environ.get('OMP_NUM_THREADS'),'estimated_peak_ram_gib':45,'available_ram_at_launch_gib':115,'test_pool':982,'test_score':'diagnostic only, not selection','engineering':'Author preprocessing/fit/predict/rescaling AST executed unchanged. Added missing pickle import, explicit paths, normalization audit. No target chunking.'}
savej(O/'STARTED.json',start)
ns={'np':np,'skl':skl,'train_latents':np.load(F/'vdvae_train_latents.npy'),'test_latents':np.load(F/'vdvae_test_latents.npy'),'train_fmri':np.load(D/'nsd_train_fmriavg_nsdgeneral_sub1.npy'),'test_fmri':np.load(D/'nsd_test_fmriavg_nsdgeneral_sub1.npy')}
body=ast.parse(source.read_text()).body
first=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and ast.unparse(n.targets[0])=='train_fmri')
last=next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and 'reg.score' in ast.unparse(n))
t0=time.time()
for node in body[first:last+1]:
 text=ast.unparse(node)
 if 'reg.fit(' in text or 'reg.predict(' in text or 'reg.score(' in text: print(json.dumps({'phase':text,'utc':now(),'elapsed_s':time.time()-t0}),flush=True)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
assert np.isfinite(ns['pred_latents']).all()
np.save(O/'nsd_vdvae_nsdgeneral_pred_sub1_31l_alpha50k.npy',ns['pred_latents'])
with open(O/'vdvae_regression_weights.pkl','wb') as f:pickle.dump({'weight':ns['reg'].coef_,'bias':ns['reg'].intercept_},f)
np.savez(O/'normalization.npz',brain_train_mean=ns['norm_mean_train'],brain_train_std_ddof1=ns['norm_scale_train'],prediction_full982_mean=np.mean(ns['pred_test_latent'],axis=0),prediction_full982_std_ddof0=np.std(ns['pred_test_latent'],axis=0),train_latent_mean=np.mean(ns['train_latents'],axis=0),train_latent_std_ddof0=np.std(ns['train_latents'],axis=0))
result={**start,'complete_utc':now(),'elapsed_seconds':time.time()-t0,'exit_status':0,'prediction_shape':list(ns['pred_latents'].shape),'prediction_dtype':str(ns['pred_latents'].dtype),'weight_shape':list(ns['reg'].coef_.shape),'brain_scale_divisor':300,'brain_std_ddof':1,'calibration':'All 982 test predictions zscore ddof0 then train latent mean/std; never T100-only calibration','files':{p.name:sha(p) for p in O.iterdir() if p.suffix in ['.npy','.npz','.pkl']}}
savej(O/'COMPLETE.json',result);print(json.dumps(result),flush=True)
