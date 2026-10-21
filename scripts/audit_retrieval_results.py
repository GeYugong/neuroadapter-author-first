"""Independently audit saved retrieval matrices and assemble the two result tables."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
from pathlib import Path
import json,csv,time
import numpy as np
from retrieval_common import R,O,sha,write
assert not (O/'final-audit.json').exists()
required=['reconstruction-recovery1-exit.json','native-bd-exit.json','native-mindeye1_s1-exit.json',
 'native-mindeye1_s2-exit.json','native-mindbridge_s1-exit.json','mindhier-native-exit.json','mindhier-images-exit.json']
for name in required:assert json.loads((O/name).read_text())['returncode']==0,name
plan=json.loads((O/'plan.json').read_text())
for p,h in plan['source_sha256'].items():assert sha(p)==h,p
for p,h in plan['ids_and_draws_sha256'].items():assert sha(O/p)==h,p
def manual_scores(sub):
 n=len(sub);row=0;col=0
 for i in range(n):
  if max(range(n),key=lambda j:sub[i,j])==i:row+=1
  if max(range(n),key=lambda j:sub[j,i])==i:col+=1
 return np.array([row/n*100,col/n*100])
rows=[];verified=[];means=[]
for p in sorted(O.glob('*/**/result.json')):
 d=json.loads(p.read_text())
 assert 'neuroadapter' not in d['method'].lower()
 s=d['subject'];draws=np.load(O/f'draws-s{s}.npy')
 if 'per_round' in d:
  matrix=p.parent/'cosine.npy';a=np.load(matrix)
  values=np.array([[x['row_query_top1_percent'],x['column_query_top1_percent']] for x in d['per_round']])
  independently=[]
  for ix in draws:
   independently.append(manual_scores(a[np.ix_(ix,ix)]))
  assert np.allclose(values,independently,rtol=0,atol=1e-12),p
  assert np.allclose(values.mean(0),d['mean_percent'],rtol=0,atol=1e-12),p
  assert np.allclose(values.std(0,ddof=1),d['draw_sd_percent'],rtol=0,atol=1e-12)
  verified.append(dict(path=str(p.relative_to(O)),sha256=sha(p),matrix_sha256=sha(matrix),trials=30))
 else:
  values=np.asarray(d['per_round_percent'])
  assert np.allclose(values.mean(0),d['mean_percent'],rtol=0,atol=1e-12)
  folder=p.parent.parent
  candidate_files=sorted(folder.glob('candidate[0-9]*/result.json'))
  if d['method']=='mindhier_s1':candidate_files=sorted(folder.glob('n4_candidate*/result.json'))
  assert len(candidate_files)==d['candidate_count'],(p,len(candidate_files))
  candidates=[]
  for q in candidate_files:
   v=json.loads(q.read_text())['per_round']
   candidates.append([[x['row_query_top1_percent'],x['column_query_top1_percent']] for x in v])
  assert np.allclose(np.mean(candidates,0),values,rtol=0,atol=1e-12),p
  means.append(str(p.relative_to(O)))
 variant=d.get('variant',p.parent.name)
 rows.append(dict(subject=s,method=d['method'],variant=variant,kind=d['kind'],
  row_direction=d['direction_names'][0],column_direction=d['direction_names'][1],
  row_top1_percent=d['mean_percent'][0],column_top1_percent=d['mean_percent'][1],
  row_draw_sd_pp=d['draw_sd_percent'][0],column_draw_sd_pp=d['draw_sd_percent'][1],
  GT_assisted=d.get('GT_assisted',False),n=982,candidates=300,repeats=30,source=str(p.relative_to(O))))
expected_images={'mindeye1_s1','mindbridge_s1','takagi_s1','brain_diffuser_s1','imagenet_s1','mindeye1_s2','mindhier_s1'}
expected_native={'mindeye1_s1','mindeye1_s2','mindbridge_s1','brain_diffuser_s1','mindhier_s1'}
assert {x['method'] for x in rows if x['kind']=='reconstruction'}==expected_images
assert {x['method'] for x in rows if x['kind']=='native'}==expected_native
for kind in ['reconstruction','native']:
 selected=[x for x in rows if x['kind']==kind]
 with (O/f'{kind}-all-variants.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
primary={
 'mindeye1_s1':{'candidate0','native16'},
 'mindbridge_s1':{'candidate0','native8_GT_assisted'},
 'takagi_s1':{'candidate0','candidate_mean'},
 'brain_diffuser_s1':{'candidate0'},
 'imagenet_s1':{'candidate0'},
 'mindeye1_s2':{'K1','native16_selected'},
 'mindhier_s1':{'n1_candidate0','N4_GT_assisted'}}
main=[x for x in rows if x['kind']=='reconstruction' and x['variant'] in primary[x['method']]]
assert len(main)==12,len(main)
with (O/'reconstruction-primary.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(main[0]));w.writeheader();w.writerows(main)
write(O/'native-applicability.json',dict(evaluated=sorted(expected_native),
 not_applicable={'takagi_s1':'Existing regression predicts SD latent/text conditioning, not the matched CLIP image-token space',
                 'imagenet_s1':'Encoding-model retrieval baseline; its native brain-space scoring is not the paper CLIP-decoding protocol'},
 excluded=['NeuroAdapter S1','NeuroAdapter S2','NeuroAdapter B joint E75','UMBRAE caption-only']))
write(O/'summary.json',dict(complete=True,image_primary=main,native=[x for x in rows if x['kind']=='native'],
 sampling='Same saved30x300 paired subsets per subject; common982IDs; MindHier original1000 restricted to the same982',
 precision='Common reconstruction features FP32, same OpenAI CLIP-L/14 checkpoint/preprocessing; FP64 cosine; native spaces remain model-specific',
 limitations=['Reconstruction retrieval and native brain-feature retrieval are different measures and are tabulated separately',
 'GT-assisted selected rows use preexisting GT-guided selections; no new selection was performed',
 'MindEye1S1 official pretrained; S2 locally trained; no retraining or checkpoint switch',
 'FP32/FP64 precision,982IDpool and subject coverage differ from MindEye2 paper; not exact paper reproduction',
 'SD across30 overlapping pools reflects sampling variability, not30 independent experiments',
 'Native Brain-Diffuser preserves author full-test prediction-distribution calibration',
 'MindHier ran on5090; common CLIP first4 GT feature cross-host agreement was checked; no bitwise hardware-equivalence claim'],
 timestamp=time.time()))
write(O/'final-audit.json',dict(complete=True,independent_python_argmax_checks=len(verified)*30*300*2,
 verified_matrix_results=verified,candidate_mean_audits=means,primary_image_rows=len(main),
 native_rows=5,image_method_subject_groups=7,source_and_sampling_hashes_valid=True,
 old_models_and_generated_images_read_only=True,new_training=0,new_generation=0,
 excluded_neuroadapter_variants=True,time=time.time()))
print('FINAL_RETRIEVAL_AUDIT_PASSED',len(verified),'matrices',len(means),'candidate means',flush=True)
