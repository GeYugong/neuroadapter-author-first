from pathlib import Path
import requests,zipfile,json,hashlib
B=Path('/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1');O=B/'assets/coco-official';O.mkdir(exist_ok=True)
f=O/'annotations_trainval2017.zip'
assert not f.exists()
s=requests.Session();s.trust_env=False
url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
r=s.get(url,stream=True,timeout=90);r.raise_for_status()
with f.with_suffix('.part').open('wb') as out:
 for x in r.iter_content(8388608):out.write(x)
f.with_suffix('.part').rename(f)
with zipfile.ZipFile(f) as z:
 for name in ['annotations/captions_train2017.json','annotations/captions_val2017.json']:
  (O/Path(name).name).write_bytes(z.read(name))
(O/'COMPLETE.json').write_text(json.dumps({'url':url,'zip_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size,'purpose':'Takagi original NSDAccess caption order; no curated substitution'},indent=2))
print('COMPLETE',flush=True)