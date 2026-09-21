"""Compare the two authorized short real-data trajectories, not model quality."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
a=root/'runs/smoke-continuous-v2'; b=root/'runs/smoke-split'
report={'rtol':1e-6,'atol':1e-7,'max_abs_error':0.0,'tensors':0,'rng_exact':True,'sample_order_exact':True}
def compare(x,y,path,exact=False):
    if isinstance(x,torch.Tensor):
        report['tensors']+=1
        if not torch.equal(x,y):
            if exact or not x.is_floating_point():raise AssertionError(path)
            error=float((x-y).abs().max())
            report['max_abs_error']=max(report['max_abs_error'],error)
            assert torch.allclose(x,y,rtol=report['rtol'],atol=report['atol']), (path,error)
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
    elif isinstance(x,dict):
        assert x.keys()==y.keys(),path
        for k in x:compare(x[k],y[k],path+'/'+str(k),exact)
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y),path
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i),exact)
    else:assert x==y,(path,x,y)
for epoch in [1,2]:
    assert (a/f'order-{epoch}.json').read_text()==(b/f'order-{epoch}.json').read_text()
    pa=a/f'checkpoint-{epoch}';pb=b/f'checkpoint-{epoch}'
    for name in ['pytorch_model.bin','pytorch_model_1.bin','optimizer.bin','random_states_0.pkl','random_states_1.pkl']:
        x=torch.load(pa/name,map_location='cpu',weights_only=False);y=torch.load(pb/name,map_location='cpu',weights_only=False)
        compare(x,y,f'{epoch}/{name}',exact=name.startswith('random'))
        del x,y
    x=json.loads((pa/'progress.json').read_text());y=json.loads((pb/'progress.json').read_text())
    for k in ['completed_epochs','global_optimizer_updates','actual_samples_processed','skipped_updates','current_lr']:assert x[k]==y[k]
audit=json.loads((a/'update_audit.json').read_text())
params=audit['before']['parameters'];changed=[]
for key,value in params.items():
    assert value['has_grad'],key
    if value['sha256']!=audit['after']['parameters'][key]['sha256']:changed.append(key)
assert any(x.startswith('guidance.') for x in changed)
assert any('image_proj_model' in x for x in changed)
assert any('to_k_ip' in x for x in changed) and any('to_v_ip' in x for x in changed)
report.update(updated_parameters=changed,max_memory_allocated=audit['after']['max_memory_allocated'],status='passed',
              note='Fresh process evaluation occurred between split epochs; short-test weights are never formal initialization.')
(root/'reports/resume_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='updated_parameters'},indent=2))
