"""Verify the delivered scores against BrainHub's saved output and inference records."""
from pathlib import Path
import ast,hashlib,json,re,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'reports/umbrae-subject01-20261021-4090'
sys.path[:0]=[str(R/'scripts'),str(R/'runtime/umbrae/upstream/UMBRAE/umbrae')]
from run_umbrae_official_score import NAMES
from transformers import LlamaTokenizer
import utils
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
final=json.loads((O/'final-metrics.json').read_text());delivery=json.loads((O/'delivery-audit.json').read_text())
text=R/final['raw_output'];assert sha(text)==final['raw_output_sha256']
values={}
for line in text.read_text().splitlines():
    m=re.fullmatch(r'([A-Za-z0-9-]+): (-?[0-9]+\.[0-9]+)',line)
    if m and m[1] in NAMES:values[NAMES[m[1]]]=float(m[2])
assert values==final['raw'] and len(values)==13 and np.isfinite(list(values.values())).all()
assert sha(R/'runtime/umbrae/upstream/BrainHub/eval_caption.py')==final['source_sha256']
plan=json.loads((O/'plan.json').read_text());snapshot=R/delivery['inference_source_snapshot']
assert sha(snapshot)==delivery['inference_source_sha256']==plan['source_hashes']['scripts/umbrae_subject01_4090.py']
old=ast.parse(snapshot.read_text());new=ast.parse((R/'scripts/umbrae_subject01_4090.py').read_text())
for name in ['models','features','caption','preflight','generate','materialize_shards','wait_for_gpu0','release_shards','run']:
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b)
mapping=json.loads((O/'input-index.json').read_text());captions=json.loads((O/'captions.json').read_text())
per=json.loads((R/final['per_instance']).read_text());frame=pd.read_csv(O/'per-sample-final.csv')
assert set(captions)==set(per)=={str(i) for i in range(982)}
assert len(list((O/'records').glob('*.json')))==982 and frame.native_index.tolist()==list(range(982))
tok=LlamaTokenizer.from_pretrained(str(R/'runtime/umbrae/assets/official/model_weights/shikra-7b'),local_files_only=True)
h=hashlib.sha256()
for i,row in enumerate(mapping):
    p=O/'records'/f'{i:04}.json';rec=json.loads(p.read_text());h.update(p.read_bytes())
    assert rec['native_index']==i and rec['nsd_id0']==row['nsd_id0']==int(frame.iloc[i].nsd_id0)
    assert not rec['GT_used'] and rec['seed']==42
    raw=tok.batch_decode([rec['token_ids']])[0]
    assert raw==rec['raw_response'] and utils.extract_id_bbox_caption(raw)[1]==rec['caption']==captions[str(i)]==frame.iloc[i].caption
    assert sha(O/'evaluation-images'/f'{i}.png')==row['evaluation_png_sha256']
    for k,v in per[str(i)].items():assert np.isfinite(v) and abs(float(frame.iloc[i][k])-v)<1e-12
assert h.hexdigest()==delivery['records_digest']
assert json.loads((O/'preflight-0.json').read_text())['original_loop_tokens_exact']
stream=json.loads((O/'stream-audit.json').read_text());assert stream['all_three_verified'] and stream['projector_matches_Shikra']
result=dict(passed=True,n=982,metrics=13,published_scores_match_original_CLI=True,all_caption_tokens_IDs_and_images_verified=True,
    official_per_instance_values_preserved=True,inference_source_snapshot_verified=True,source_generation_functions_unchanged=True,
    official_evaluator_sha256=final['source_sha256'],final_metrics_sha256=sha(O/'final-metrics.json'))
p=O/'verification.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
