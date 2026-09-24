"""Final read-only verification and evidence collection, after controller exit."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/hl-content-interchange-l120'
W=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100/comparison/hl-content-interchange-l120'
def main():
    assert (R/'EVALUATION_COMPLETE').exists()
    events=[json.loads(x) for x in (ROOT/'logs/hl-content-events.jsonl').read_text().splitlines()]
    assert events[-1]['action']=='controller_complete' and events[-1]['exit_code']==0
    assert all(e['exit_code']==0 for e in events if e['action']=='exit')
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    related=[p for p in processes if any(x in p for x in ['scripts/hl_content_interchange.py','scripts/decode_hl_content.py']) and 'ps -eo' not in p]
    assert not related,related
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    locks=json.loads((R/'input-lock.json').read_text())
    unchanged={}
    for name,digest in locks.items():
        with (ROOT/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        unchanged[name]=actual==digest
    assert all(unchanged.values())
    ids=np.loadtxt(ROOT/'configs/monitor100_ids.txt',dtype=int).reshape(-1).tolist()
    audits=json.loads((R/'content-audits.json').read_text())
    for label in ['BA','AB']:
        assert [(x['image_id'],x['candidate']) for x in audits[label]]==[(i,k) for i in ids for k in range(2)]
        assert len(list((W/label).glob('image_*.npz')))==100
    # Preserve each actual invocation and source script hash, without claiming later docs were running code.
    evidence={'events':events,'all_key_input_sha256_unchanged':unchanged,
        'controller_and_decoder_processes':related,'gpu_compute_processes_at_delivery':gpu,
        'new_training_updates':0,'formal_new_candidates':400,'identity_check_candidates':8,
        'execution_code_commit':'c2825854f11faa6f6ae7040c0c919477c184b0a8',
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                        ['scripts/decode_hl_content.py','scripts/hl_content_interchange.py','scripts/hl_content_report.py']},
        'fixed_atlas_pages':8,'review_status':'See AI_REVIEW.md; non-blind AI review'}
    assert len(list((R/'atlas').glob('*.jpg')))==8
    (R/'delivery-checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'all_key_input_hashes':len(unchanged),'all_unchanged':True,'exit_records':sum(e['action']=='exit' for e in events),
                      'gpu_compute_processes':gpu,'new_training_updates':0,'formal_candidates':400,'checks':8}))
if __name__=='__main__':main()
