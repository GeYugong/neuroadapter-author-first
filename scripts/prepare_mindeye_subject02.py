"""Audit the official Subject2 shards and create a bounded indexed validation view."""
import json
from pathlib import Path
import time
from mindeye_subject02_common import R,O,D,S,sha,write,index_tar


def main():
    plan=json.loads((O/'asset-plan.json').read_text());rows={'train':[],'val':[]}
    receipts=[]
    for index,item in enumerate(plan['files']):
        receipt=O/'downloads'/f'{index:02d}.json'
        assert receipt.exists(),f'Asset {index} not ready'
        known=json.loads(receipt.read_text());target=D/item['path']
        assert known['verified'] and target.stat().st_size==item['bytes'] and sha(target)==known['actual_sha256']
        if item['sha256']:assert known['actual_sha256']==item['sha256']
        receipts.append({'path':item['path'],'sha256':known['actual_sha256'],'bytes':item['bytes']})
        if item['path'].endswith('.tar'):
            split=Path(item['path']).parent.name
            values=index_tar(target);rows[split].extend(values)
            print('INDEXED',target.name,len(values),flush=True)
    test=set(int(x) for x in (R/'data/subject02/standard-test-ids.txt').read_text().split())
    train={x['image_id'] for x in rows['train']};val={x['image_id'] for x in rows['val']}
    assert len(rows['train'])==len(train)==8559 and len(rows['val'])==len(val)==300
    assert len(test)==982 and not train&val and not test&(train|val)
    write(O/'data-index.json',rows)
    write(O/'data-ready.json',{'complete':True,'subject':2,'train':8559,'val':300,'test_excluded':982,
        'all_IDs_unique_and_disjoint':True,'original_brain_shape':[3,14278],
        'original_brain_dtype':'float16','all_brain_values_finite':True,'original_JPEG_RGB_shape':[256,256],
        'official_arrays_unchanged':True,'receipts':receipts,'data_index_sha256':sha(O/'data-index.json'),
        'time':time.time()})
    print('DATA_READY',flush=True)


if __name__=='__main__':main()
