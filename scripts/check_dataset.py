"""Check real author dataset outputs against their raw HDF5 sources."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from brain_adapter.dataset import nsd_topk_parcel_dataset
from transformers import CLIPTokenizer
c=json.loads((root/'configs/stage1.json').read_text())
tokenizer=CLIPTokenizer.from_pretrained(c['pretrained_model_name_or_path'],subfolder='tokenizer')
a=SimpleNamespace(subj=1,hemi=None,backbone_arch='dinov2_q',data_dir=c['data_dir'],imgs_dir=c['imgs_dir'],parcel_dir=c['parcel_dir'],tokenizer=tokenizer,gen_size=512)
d=nsd_topk_parcel_dataset(a,'train',topk=100)
v=nsd_topk_parcel_dataset(a,'val',topk=100)
assert d.max_voxels==v.max_voxels==626
for h in ['lh','rh']:
    assert np.array_equal(d.selected_parcel_idx[h],v.selected_parcel_idx[h])
    for x,y in zip(d.get_selected_voxel_indices()[h],v.get_selected_voxel_indices()[h]):assert torch.equal(x,y)
rows=[]
for image_id in np.loadtxt(root/'configs/smoke16_ids.txt',dtype=int):
    idx=int(np.where(d.base_dataset.split_presented_imgs==image_id)[0][0])
    t=time.perf_counter();b=d[idx];seconds=time.perf_counter()-t
    runs=d.base_dataset.img_to_runs[idx]
    assert len(runs)==3 and np.all(d.base_dataset.img_order[runs]==image_id)
    raw=d.base_dataset.imgs['imgBrick'][image_id]
    assert torch.equal(b['img_encoder'],torch.from_numpy(raw).permute(2,0,1).float()/255)
    assert torch.equal(b['img_ipadapter'],d.ipadapter_transform(raw))
    for j,h in enumerate(['lh','rh']):
        mean=torch.from_numpy(d.base_dataset.betas[j][runs]).mean(0)
        for k,verts in enumerate(d.get_selected_voxel_indices()[h]):
            assert torch.equal(b[f'brain_{h}_f'][k,:len(verts)],mean[verts])
            assert torch.count_nonzero(b[f'brain_{h}_f'][k,len(verts):])==0
        assert torch.isfinite(b[f'brain_{h}_f']).all()
    rows.append(dict(image_id=int(image_id),presentations=runs.tolist(),seconds=seconds,brain_shape=[200,626],image_shape=list(b['img_ipadapter'].shape)))
report={'samples':rows,'parcel_identity_shared':True,'mean_read_seconds':float(np.mean([x['seconds'] for x in rows]))}
(root/'reports/dataset_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
