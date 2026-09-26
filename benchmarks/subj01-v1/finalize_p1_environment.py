import pathlib,json,subprocess,datetime,hashlib
B=pathlib.Path(__file__).resolve().parent
p=B/'probe_p1_imports.py'
s=p.read_text();s=s.replace("['lib.model_zoo.vd','lib.model_zoo.ddim_vd']", "['lib.model_zoo.vd','lib.model_zoo.ddim_vd','lib.model_zoo.optimus','lib.model_zoo.clip','lib.model_zoo.autoencoder']")
p.write_text(s)
P=B/'envs/p1-author310/bin/python'
freeze=subprocess.check_output([str(P),'-m','pip','freeze'],text=True)
(B/'p1-environment-freeze.txt').write_text(freeze)
a={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':str(P.parent.parent),'scope':'CPU import only; no model construction, inference, regression, or GPU','base_read_only':str(B/'envs/mindeye1/lib/python3.10/site-packages'),'local_packages':'pytorch-lightning==1.4.2 torchmetrics==0.6.0 pyDeprecate==0.3.1 omegaconf==2.1.1 antlr4-python3-runtime==4.8 taming-transformers-rom1504==0.0.6 tensorboard==2.12.3 absl-py==1.4.0 grpcio==1.59.3 markdown==3.5.1 tensorboard-data-server==0.7.2 werkzeug==2.3.8','engineering_notes':['Python3.10.12 torch2.0.1+cu117 reused read-only from MindEye1; author original Brain-Diffuser Python3.8 torch1.12.1 and Takagi torch1.11 differ. Runtime numerical equivalence not claimed by import success.','No author source changes. Local environment only missing import dependencies installed --target --no-deps.','Initial runtime-created p1-author Python3.11 is unused; incompatible ABI with read-only Python3.10 packages. Active environment is p1-author310.','System Python3.10 lacks ensurepip; created --without-pip then read-only inherited pip, all writes explicit --target active environment.'],'upstream_commits':{k:subprocess.check_output(['git','-C',str(B/'upstream'/k),'rev-parse','HEAD'],text=True).strip() for k in ['brain-diffuser','takagi']}}
(B/'p1-environment-audit.json').write_text(json.dumps(a,indent=2))
