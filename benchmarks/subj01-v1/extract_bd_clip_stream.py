"""Stream author Brain-Diffuser frozen CLIP features; no generator training."""
from pathlib import Path
import os,sys,json,hashlib,types
import numpy as np
import h5py,torch
from PIL import Image
import torchvision.transforms as T
B=Path(__file__).resolve().parent
U=B/'upstream/brain-diffuser/versatile_diffusion'
sys.path.insert(0,str(U));os.chdir(U)
from lib.model_zoo.clip import FrozenCLIP
from lib.model_zoo.vd import VD
OUT=B/'processed/brain-diffuser-features';OUT.mkdir(parents=True,exist_ok=True)
lock=OUT/'clip-extraction-start.json'
assert not lock.exists(),'Existing extraction: inspect before resuming'
lock.write_text(json.dumps({'policy':'author FrozenCLIP and VD.clip_encode_* unchanged; load exact clip.* subset from full VD checkpoint; image512 batch1 float64 outputs; test then train; no gradients'},indent=2))
clip=FrozenCLIP(version=str(B/'assets/clip-vit-large-patch14'))
weights=B/'assets/p1-models/brain-diffuser/versatile_diffusion/vd-four-flow-v1-0-fp16-deprecated.pth'
sd=torch.load(weights,map_location='cpu')
sub={k[5:]:v for k,v in sd.items() if k.startswith('clip.')}
assert sub,'Missing clip prefix'
clip.load_state_dict(sub,strict=True)
assert all(torch.equal(v,sub[k]) for k,v in clip.state_dict().items())
del sub,sd
clip=clip.cuda()
net=types.SimpleNamespace(clip=clip)
for name in ['clip_encode_text','clip_encode_vision']:setattr(net,name,types.MethodType(getattr(VD,name),net))
(OUT/'clip-load-audit.json').write_text(json.dumps({'strict':True,'all_tensors_equal':True,'class':'author FrozenCLIP','methods':'author VD.clip_encode_text/vision','unused_generator_modules':'not instantiated for feature extraction only'},indent=2))
brain=B/'processed/regression-brain'
caps=np.load(B/'assets/p1-models/brain-diffuser/COCO_73k_annots_curated.npy',mmap_mode='r')
raw=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/stimuli/nsd/nsd_stimuli.hdf5')
if not raw.exists():raw=Path('/data1/matengyu/geyugong/neuroadapter-subject1-research/data/raw/nsd/stimuli/nsd_stimuli.hdf5')
assert raw.exists(),raw
with h5py.File(raw,'r') as hf,torch.no_grad():
 for split in ['test','train']:
  ids=np.load(brain/f'{split}_unique_ids.npy')
  outputs={name:np.lib.format.open_memmap(OUT/f'nsd_clip{name}_{split}.npy',mode='w+',dtype=np.float64,shape=(len(ids),n,768)) for name,n in [('vision',257),('text',77)]}
  for idx,image_id in enumerate(ids):
   im=Image.fromarray(hf['imgBrick'][int(image_id)])
   cin=T.functional.to_tensor(T.functional.resize(im,(512,512))).float()*2-1
   v=net.clip_encode_vision(cin.unsqueeze(0))[0].cpu().numpy()
   annots=caps[int(image_id)];texts=list(annots[annots!=''])
   c=net.clip_encode_text(texts).cpu().numpy().mean(0)
   assert v.shape==(257,768) and c.shape==(77,768) and np.isfinite(v).all() and np.isfinite(c).all()
   outputs['vision'][idx]=v;outputs['text'][idx]=c
   if idx%100==0:print(split,idx,len(ids),flush=True)
  for arr in outputs.values():arr.flush()
  (OUT/f'clip-{split}.done').write_text(str(len(ids)))
(OUT/'CLIP_COMPLETE.json').write_text(json.dumps({'train':8859,'test':982,'dtype':'float64','batch_size':1,'exit_status':0}))