import json,os,subprocess,pathlib
B=pathlib.Path(__file__).resolve().parent
P=B/'envs/p1-author310/bin/python'
tests=[('vdvae',B/'upstream/brain-diffuser/vdvae',['vae','model_utils']),('vd',B/'upstream/brain-diffuser/versatile_diffusion',['lib.model_zoo.vd','lib.model_zoo.ddim_vd','lib.model_zoo.optimus','lib.model_zoo.clip','lib.model_zoo.autoencoder']),('takagi_sd14',B/'upstream/takagi/codes/diffusion_sd1/stable-diffusion',['ldm.models.diffusion.ddpm','ldm.models.diffusion.ddim','ldm.modules.encoders.modules'])]
r=[]
for name,path,mods in tests:
 code='import importlib; '+ '; '.join('importlib.import_module('+repr(m)+')' for m in mods)
 p=subprocess.run([str(P),'-c',code],cwd=path,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(path)},capture_output=True,text=True)
 r.append(dict(name=name,modules=mods,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(json.dumps(r,indent=2));(B/'p1-cpu-import-audit.json').write_text(json.dumps(r,indent=2))
