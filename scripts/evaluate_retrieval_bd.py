"""Native Brain-Diffuser image-space retrieval from frozen regression outputs."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
from pathlib import Path
import json,time,traceback
import numpy as np
from retrieval_common import R,O,sha,write,evaluate
B=R/'benchmarks/subj01-v1'
dest=O/'native/brain_diffuser_s1'
try:
 assert not dest.exists()
 ids=np.load(O/'ids-s1.npy');native_ids=np.load(B/'processed/regression-brain/test_unique_ids.npy')
 assert len(native_ids)==982 and set(ids)==set(native_ids)
 lookup={int(x):i for i,x in enumerate(native_ids)};order=[lookup[int(x)] for x in ids]
 pred=B/'processed/brain-diffuser-regression/clipvision/nsd_clipvision_predtest_nsdgeneral.npy'
 target=B/'processed/brain-diffuser-features/nsd_clipvision_test.npy'
 ph,th=sha(pred),sha(target)
 a=np.load(pred,mmap_mode='r')[order];b=np.load(target,mmap_mode='r')[order]
 assert a.shape==b.shape==(982,257,768)
 r=evaluate(a,b,np.load(O/'draws-s1.npy'),dest,dict(method='brain_diffuser_s1',subject=1,kind='native',
  direction_names=['Image (brain-to-image)','Brain (image-to-brain)'],feature='Author VD CLIP vision projected257x768 tokens; native frozen saved targets',
  precision='Existing float64 saved arrays; FP64 cosine',
  transductive_note='Author predicted test features use full982 unlabeled prediction mean/std and train target rescaling',
  no_GT_selection=True,source_sha256={str(pred):ph,str(target):th,str(Path(__file__)):sha(__file__)}))
 assert sha(pred)==ph and sha(target)==th
 write(O/'native-bd-exit.json',dict(returncode=0,time=time.time()))
 print('NATIVE_BD_COMPLETE',r['mean_percent'],flush=True)
except BaseException:
 write(O/'native-bd-failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
