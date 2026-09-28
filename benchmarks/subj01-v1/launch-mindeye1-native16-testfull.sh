#!/bin/bash
set -u
cd /data1/matengyu/geyugong/neuroadapter-author-first
runtime/env/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,shutil,subprocess
b=Path('benchmarks/subj01-v1')
p=json.loads((b/'mindeye-native16-test100-publication.json').read_text())
assert p['github_atlas_pages_verified']==4 and p['atlas_images_reviewed']==32
assert p['generation_exit']==p['scoring_exit']==0 and p['n_eval']==100
for name,expected in p['atlas_sha256'].items():
 assert hashlib.sha256((b/'comparison_grid_mindeye_native16'/name).read_bytes()).hexdigest()==expected
for name in ['mindeye1-native16-testfull.log','mindeye1-native16-testfull.exit','mindeye1-native16-scorefull.log','mindeye1-native16-scorefull.exit']:
 assert not (b/'logs'/name).exists(), 'Existing full log: inspect before explicit resume'
assert not (b/'metrics/mindeye1-native16/test_full_ids').exists()
assert shutil.disk_usage(b).free > 60*1024**3
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU0 occupied'
assert len((b/'test_full_ids.txt').read_text().split())==982
assert hashlib.sha256((b/'test_full_ids.txt').read_bytes()).hexdigest()=='4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441'
print('Native16 full982 gates passed',flush=True)
PY
[ "$?" -eq 0 ] || exit 1
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN HOME=/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/cache/inference-home HF_HOME=/data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/cache/hf-inference HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/envs/mindeye1/bin/python /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/mindeye_native16.py --ids /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/test_full_ids.txt --brain-inputs /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/assets/subject1_test_brain.npz --output /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/outputs/mindeye1-native16 --upstream /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/upstream/mindeye1 --high-checkpoint /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/assets/mindeye/mindeye_models/prior_257_final_subj01_bimixco_softclip_byol/last.pth --low-checkpoint /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/assets/mindeye/mindeye_models/autoencoder_subj01_4x_locont_no_reconst/epoch120.pth --vd /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/assets/versatile-diffusion --clip-vision /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/assets/clip-vit-large-patch14 --openai-clip /data1/matengyu/geyugong/neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-native16-testfull.log 2>&1
status=$?
printf "%s\n" "$status" > /data1/matengyu/geyugong/neuroadapter-author-first/benchmarks/subj01-v1/logs/mindeye1-native16-testfull.exit
[ "$status" -eq 0 ] || exit "$status"
CUDA_VISIBLE_DEVICES=0 runtime/env/bin/python -u benchmarks/subj01-v1/evaluate_native16.py --method mindeye1-native16 --ids benchmarks/subj01-v1/test_full_ids.txt --feature-batch 16 > benchmarks/subj01-v1/logs/mindeye1-native16-scorefull.log 2>&1
status=$?
printf '%s\n' "$status" > benchmarks/subj01-v1/logs/mindeye1-native16-scorefull.exit
exit "$status"
