"""Inference-only brain-input remapping; all model and diffusion code is the original decoder."""
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np

def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def remap_fields(target, source):
    result=dict(target)
    for key in ('brain_lh_f','brain_rh_f'):
        assert target[key].shape==source[key].shape and target[key].dtype==source[key].dtype==torch.float32
        assert torch.isfinite(source[key]).all()
        result[key]=source[key]
    assert all(result[k] is target[k] for k in target if k not in ('brain_lh_f','brain_rh_f'))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-mode',choices=['identity','permuted'],required=True)
    parser.add_argument('--control-map',required=True)
    control,remaining=parser.parse_known_args()
    import decode_brain_adapter as decoder
    args=decoder.create_argument_parser().parse_args(remaining)
    assert args.no_selection and args.split=='val' and args.batch_size==1 and args.num_workers==0
    assert args.num_predictions==2 and args.denoising_steps==50 and args.noise_factor==4 and args.seed==42
    assert args.saved_epochs=='120' and args.precision=='fp32'
    run=ROOT/'runs/subject01-author-fp32-seed42-lr5e-5-from100'
    assert Path(args.export_path)==run/'exports/epoch-120/inference.pt'
    destination=Path(args.decoded_stimuli_dir).resolve()
    assert destination.is_relative_to(run/'comparison/brain-correspondence-l120')
    assert not destination.exists(), 'Refuse existing control output directory'
    full=np.loadtxt(ROOT/'configs/val_ids.txt',dtype=int).tolist()
    entries=json.loads(Path(control.control_map).read_text())
    mapping={int(row['target_image_id']):int(row['brain_source_image_id']) for row in entries}
    assert len(entries)==len(mapping)==500 and set(mapping)==set(full) and set(mapping.values())==set(full)
    assert all(i!=j for i,j in mapping.items())
    requested=np.loadtxt(args.image_ids,dtype=int).reshape(-1).tolist()
    if control.control_mode=='identity':
        assert requested==np.loadtxt(ROOT/'configs/decode2_ids.txt',dtype=int).reshape(-1).tolist()
    else: assert requested==full
    destination.mkdir(parents=True)
    pending=[];records=[]
    original_create=decoder.create_test_dataset
    original_guidance=decoder.GuidanceGenerator

    def create_dataset(*a,**kw):
        dataset,loader,indices=original_create(*a,**kw)
        id_to_idx={int(v):i for i,v in enumerate(dataset.base_dataset.split_presented_imgs)}
        assert set(id_to_idx)==set(full)
        selected=list(loader.dataset.indices)
        assert [int(dataset.base_dataset.split_presented_imgs[i]) for i in selected]==requested
        class Remapped(torch.utils.data.Dataset):
            def __len__(self):return len(selected)
            def __getitem__(self,index):
                assert not pending, 'Unconsumed conditioning audit'
                target=dataset[selected[index]]
                image_id=int(target['image_id'])
                source_id=image_id if control.control_mode=='identity' else mapping[image_id]
                source=target if source_id==image_id else dataset[id_to_idx[source_id]]
                assert int(source['image_id'])==source_id
                result=remap_fields(target,source)
                expected=torch.cat([result['brain_lh_f'],result['brain_rh_f']],dim=0).unsqueeze(0)
                original=torch.cat([target['brain_lh_f'],target['brain_rh_f']],dim=0).unsqueeze(0)
                same=torch.equal(expected,original)
                assert same if control.control_mode=='identity' else not same
                pending.append(dict(target_image_id=image_id,brain_source_image_id=source_id,
                    expected_input_sha256=tensor_sha(expected),target_input_sha256=tensor_sha(original),
                    target_gt_sha256=tensor_sha(target['img_encoder']),nonbrain_fields_unchanged=True))
                return result
        replacement=torch.utils.data.DataLoader(Remapped(),batch_size=1,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
        return dataset,replacement,indices

    def guidance(*a,**kw):
        model=original_guidance(*a,**kw)
        def before_forward(module,inputs):
            assert len(pending)==1 and len(inputs)==1
            row=pending.pop()
            assert inputs[0].dtype==torch.float32
            actual=tensor_sha(inputs[0])
            assert actual==row['expected_input_sha256']
            row['actual_generator_input_sha256']=actual
            records.append(row)
            with (destination/'brain-input-audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        model.register_forward_pre_hook(before_forward)
        return model

    decoder.create_test_dataset=create_dataset
    decoder.GuidanceGenerator=guidance
    sys.argv=[str(ROOT/'decode_brain_adapter.py')]+remaining
    decoder.main()
    assert not pending and [x['target_image_id'] for x in records]==requested
    (destination/'control-protocol.json').write_text(json.dumps(dict(mode=control.control_mode,
        map_path=str(Path(control.control_map).resolve()),map_sha256=hashlib.sha256(Path(control.control_map).read_bytes()).hexdigest(),
        intervention='Replace only brain_lh_f/brain_rh_f before original GuidanceGenerator; retain target GT/ID and target-ID candidate seeds.',
        pid=os.getpid(),input_rows_verified=len(records)),indent=2)+'\n')

if __name__=='__main__':main()
