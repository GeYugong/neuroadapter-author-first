"""Validate saved captions and compare the score adapter with the upstream CLI."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,time
import numpy as np,torch
R=Path(__file__).resolve().parents[1]
O=R/'reports/umbrae-subject01-20261021-4090'
A=R/'runtime/umbrae/assets'
B=R/'runtime/umbrae/upstream/BrainHub'
U=R/'runtime/umbrae/upstream/UMBRAE/umbrae'
assert json.loads((O/'exit.json').read_text())['returncode']==0
assert json.loads((O/'status.json').read_text())['phase']=='complete'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
torch.set_num_threads(4)
sys.path[:0]=[str(U),str(B)]
from transformers import LlamaTokenizer
import utils
tokenizer=LlamaTokenizer.from_pretrained(str(A/'official/model_weights/shikra-7b'),padding_side='left',local_files_only=True)
captions=json.loads((O/'captions.json').read_text());mapping=json.loads((O/'input-index.json').read_text())
assert set(captions)=={str(i) for i in range(982)}
records_digest=hashlib.sha256();token_counts=[]
for i,row in enumerate(mapping):
    p=O/'records'/f'{i:04}.json';rec=json.loads(p.read_text())
    assert rec['native_index']==i and rec['nsd_id0']==row['nsd_id0'] and rec['GT_used'] is False
    assert rec['seed']==42 and rec['prompt']=='Describe this image <image> as simply as possible.'
    raw=tokenizer.batch_decode([rec['token_ids']])[0]
    assert raw==rec['raw_response']
    assert utils.extract_id_bbox_caption(raw)[1]==rec['caption']==captions[str(i)]
    assert rec['caption'].strip();token_counts.append(len(rec['token_ids']))
    records_digest.update(p.read_bytes())
target=O/'native-score-check';target.mkdir(exist_ok=False)
(target/'captions.json').symlink_to(O/'captions.json')
spec=importlib.util.spec_from_file_location('umbrae_brainhub_reference',B/'eval_caption.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
original_listdir=native.os.listdir
def canonical_images(path):
    if Path(path)==O/'evaluation-images':return [f'{i}.png' for i in range(982)]
    return original_listdir(path)
native.os.listdir=canonical_images
original_clip_load=native.clip.load
def local_clip(name,*args,**kwargs):
    assert name=='ViT-B/32'
    return original_clip_load(str(A/'evaluation/ViT-B-32.pt'),*args,**kwargs)
native.clip.load=local_clip
original_sentence=native.SentenceTransformer
def local_sentence(name,*args,**kwargs):
    assert name=='all-MiniLM-L6-v2'
    return original_sentence(str(A/'minilm'),*args,**kwargs)
native.SentenceTransformer=local_sentence
native._MODELS={'ViT-B/32':str(A/'evaluation-extras/clip_ViT-B-32.pth')}
captured={};original_core=native.get_all_metrics
def capture_core(*args,**kwargs):
    result=original_core(*args,**kwargs);captured.update(result);return result
native.get_all_metrics=capture_core
sys.argv=['eval_caption.py',str(target/'captions.json'),str(O/'evaluation-images'),
          '--references_json',str(B/'data/caption/fmri_cococap.json'),'--save_per_instance',str(target/'per-instance.json')]
native.main()
per=json.loads((target/'per-instance.json').read_text());assert set(per)==set(captions)
reference={f'BLEU-{i+1}':float(x) for i,x in enumerate(captured['bleu'])}
reference.update({name:float(captured[key]) for name,key in [('METEOR','meteor'),('ROUGE-L','rouge'),('CIDEr','cider'),('SPICE','spice')]})
for key in ['CLIPScore','RefCLIPScore','PACScore','RefPACScore','SentenceScore']:
    reference[key]=float(np.mean([per[str(i)][key] for i in range(982)]))
primary=json.loads((O/'metrics.json').read_text())['raw']
errors={k:abs(reference[k]-primary[k]) for k in primary}
comparison=O/'native-score-check/comparison.json'
assert not comparison.exists()
comparison.write_text(json.dumps(dict(reference_scores=reference,adapter_scores=primary,absolute_errors=errors),indent=2)+'\n')
assert all(np.isfinite(v) and v<=1e-6 for v in errors.values()),errors
audit=dict(complete=True,n=982,all_saved_tokens_decode_exactly=True,all_caption_IDs_and_prompts_verified=True,
    original_CLI_parity=True,max_abs_error=max(errors.values()),metric_errors=errors,reference_scores=reference,
    reference_adaptations='Local asset paths, canonical test-ID enumeration and captured return values; scoring source unmodified',
    records_digest=records_digest.hexdigest(),token_count=dict(min=min(token_counts),max=max(token_counts),mean=float(np.mean(token_counts))),time=time.time())
p=O/'delivery-audit.json';assert not p.exists();p.write_text(json.dumps(audit,indent=2)+'\n')
print('DELIVERY_AUDIT_PASSED',json.dumps(errors))
