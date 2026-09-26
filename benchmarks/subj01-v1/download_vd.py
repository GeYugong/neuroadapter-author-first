from pathlib import Path
import json,hashlib
from huggingface_hub import hf_hub_download
B=Path(__file__).resolve().parent
lock=json.loads((B/'vd-download-lock.json').read_text())
result=[]
for f in lock['files']:
    if f['path'].startswith(('pretrained_pth/','assets/','text_unet/')):continue
    print('START',f['path'],flush=True)
    p=Path(hf_hub_download(lock['repo'],f['path'],revision=lock['revision'],local_dir=B/'assets/versatile-diffusion',token=False))
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
    assert f['sha256'] is None or h.hexdigest()==f['sha256']
    result.append(dict(f,actual_sha256=h.hexdigest()))
    print('VERIFIED',f['path'],flush=True)
(B/'vd-download-complete.json').write_text(json.dumps(result,indent=2))
