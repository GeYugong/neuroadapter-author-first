"""Run BrainHub's evaluation entry with local assets and save its printed scores."""
from pathlib import Path
import hashlib,importlib.util,json,os,re,sys,time
R=Path(__file__).resolve().parents[1]
O=R/'reports/umbrae-subject01-20261021-4090'
A=R/'runtime/umbrae/assets'
B=R/'runtime/umbrae/upstream/BrainHub'
NAMES={'BLEU-1':'BLEU-1','BLEU-2':'BLEU-2','BLEU-3':'BLEU-3','BLEU-4':'BLEU-4',
       'METEOR':'METEOR','ROUGE':'ROUGE-L','CIDER':'CIDEr','SPICE':'SPICE',
       'CLIPScore':'CLIPScore','RefCLIPScore':'RefCLIPScore','PACScore':'PACScore',
       'RefPACScore':'RefPACScore','SentenceScore':'SentenceScore'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finalize(directory):
    text=(directory/'results_.txt').read_text();scores={}
    for line in text.splitlines():
        match=re.fullmatch(r'([A-Za-z0-9-]+): (-?[0-9]+\.[0-9]+)',line)
        if match and match[1] in NAMES:
            key=NAMES[match[1]];assert key not in scores;scores[key]=float(match[2])
    assert set(scores)==set(NAMES.values()) and len(scores)==13
    per=json.loads((directory/'per-instance.json').read_text())
    assert set(per)=={str(i) for i in range(982)}
    result=dict(n=982,raw=scores,display_times100={k:v*100 for k,v in scores.items()},
        precision='Original CLI output rounded to4 decimal places',
        score_source='Official BrainHub eval_caption.py main entry; no custom score formulas or aggregation',
        source_sha256=digest(B/'eval_caption.py'),raw_output=str((directory/'results_.txt').relative_to(R)),
        raw_output_sha256=digest(directory/'results_.txt'),per_instance=str((directory/'per-instance.json').relative_to(R)),
        inference='Official UMBRAE brainx-v-1-4 cross-subject checkpoint evaluated on S1',time=time.time())
    target=O/'final-metrics.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    return result
def score():
    import torch
    torch.set_num_threads(4)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    target=O/'official-evaluation';target.mkdir(exist_ok=False)
    rows=json.loads((O/'input-index.json').read_text());captions={}
    for i,row in enumerate(rows):
        rec=json.loads((O/'records'/f'{i:04}.json').read_text())
        assert rec['native_index']==i and rec['nsd_id0']==row['nsd_id0'] and rec['GT_used'] is False
        captions[str(i)]=rec['caption']
    assert len(captions)==982
    if not (O/'captions.json').exists():
        (O/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    else:assert json.loads((O/'captions.json').read_text())==captions
    (target/'captions.json').symlink_to(O/'captions.json')
    sys.path.insert(0,str(B))
    spec=importlib.util.spec_from_file_location('brainhub_official',B/'eval_caption.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    load_clip=native.clip.load
    def local_clip(name,*args,**kwargs):
        assert name=='ViT-B/32';return load_clip(str(A/'evaluation/ViT-B-32.pt'),*args,**kwargs)
    native.clip.load=local_clip
    sentence=native.SentenceTransformer
    def local_sentence(name,*args,**kwargs):
        assert name=='all-MiniLM-L6-v2';return sentence(str(A/'minilm'),*args,**kwargs)
    native.SentenceTransformer=local_sentence
    native._MODELS={'ViT-B/32':str(A/'evaluation-extras/clip_ViT-B-32.pth')}
    listdir=native.os.listdir
    def images(path):
        return [f'{i}.png' for i in range(982)] if Path(path)==O/'evaluation-images' else listdir(path)
    native.os.listdir=images
    sys.argv=['eval_caption.py',str(target/'captions.json'),str(O/'evaluation-images'),
        '--references_json',str(B/'data/caption/fmri_cococap.json'),'--save_per_instance',str(target/'per-instance.json')]
    native.main()
    return finalize(target)
if __name__=='__main__':print(json.dumps(score(),indent=2))
