"""Strict configuration boundary for the one authorized seed43 endpoint evaluation."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/subject01-author-fp32-seed43-lrhalf-at100'
REPORT=ROOT/'reports/seed43-hl-replication/evaluation'
WORK=RUN/'comparison/hl-content-replication'
STUDY='seed43-HL-content-replication'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def validate(path,args,condition,raw=False):
    path=Path(path).resolve();assert path.is_relative_to(REPORT)
    cfg=json.loads(path.read_text());assert cfg['study']==STUDY
    assert cfg['mode'] in ('check','formal')
    check=cfg['mode']=='check';assert not raw or check
    assert cfg['allowed_conditions']==(['AA','BB'] if check else (['AA'] if cfg['batch']==0 else ['BA','AB','BB']))
    assert condition in cfg['allowed_conditions']
    assert Path(args.export_path).resolve()==RUN/'exports/epoch-120/inference.pt'
    assert Path(args.model_weights_dir).resolve()==RUN
    meta=json.loads((RUN/'exports/epoch-120/metadata.json').read_text())
    assert meta['epoch']==120 and meta['global_optimizer_updates']==63840
    assert meta['config']['seed']==43 and meta['config']['run_id']==RUN.name
    assert sha(RUN/'exports/epoch-120/inference.pt')==meta['sha256']==cfg['model_sha256']
    locked=json.loads((ROOT/'reports/evidence-freeze-seed43-plan/protocol-proposal.json').read_text())['target_donor_and_atlas_files_sha256']
    original=ROOT/'reports/hl-content-replication-l120'
    for p in [original/'target_ids.txt',original/'atlas32_ids.txt',original/f"donor-{max(cfg['batch'],1)}/map.json",ROOT/'reports/ibbi-roi-l120-pilot/roi_mapping.json']:
        assert sha(p)==locked[str(p.relative_to(ROOT))]
    target=np.loadtxt(original/'target_ids.txt',dtype=int).reshape(-1).tolist()
    ids=np.loadtxt(args.image_ids,dtype=int).reshape(-1).tolist()
    assert Path(args.image_ids).resolve()==Path(cfg['target_ids']).resolve()
    assert ids==(target[:1] if check else target)
    expected_map=original/f"donor-{max(cfg['batch'],1)}/map.json"
    assert Path(cfg['mapping_path']).resolve()==expected_map and sha(expected_map)==cfg['mapping_sha256']
    mapping={r['target_image_id']:r['brain_source_image_id'] for r in json.loads(expected_map.read_text())}
    assert len(mapping)==len(set(mapping.values()))==100 and set(mapping)==set(target) and not set(mapping)&set(mapping.values())
    expected_root=WORK/('checks/reference' if raw else 'checks/content') if check else (WORK if cfg['batch']==0 else WORK/f"donor-{cfg['batch']}")
    out=Path(args.decoded_stimuli_dir).resolve()
    assert out==expected_root/condition and not out.exists()
    if check and not raw:assert Path(cfg['reference_root']).resolve()==WORK/'checks/reference'
    assert len(ids)*2==(2 if check else 200)
    return cfg,mapping,ids
