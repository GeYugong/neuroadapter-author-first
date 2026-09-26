"""Execute existing decoder with exact, audited K1-only entry changes."""
from pathlib import Path
import sys,hashlib,json
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'decode_brain_adapter.py';text=source.read_text()
patches={
 'assert args.batch_size == 1 and args.num_predictions == 2':'assert args.batch_size == 1 and args.num_predictions == 1',
 'for candidate in range(2):':'for candidate in range(1):',
 'candidate_batch=1,num_predictions=2,denoising_steps=':'candidate_batch=1,num_predictions=1,denoising_steps=',
}
original_sha=hashlib.sha256(text.encode()).hexdigest()
for old,new in patches.items():
 assert text.count(old)==1,old
 text=text.replace(old,new)
print(json.dumps({'original_decoder_sha256':original_sha,'patches':patches}),flush=True)
sys.path.insert(0,str(ROOT))
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
