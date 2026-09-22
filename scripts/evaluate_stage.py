"""Stage orchestration, author evaluator calls and fixed contact sheets only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['TORCH_HOME']=str(ROOT/'runtime/torch')
os.environ['AUTHOR_SWAV_SOURCE']=str(ROOT/'runtime/swav')
os.environ['AUTHOR_CLIP_PATH']=str(ROOT.parent/'neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt')
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'


def decode(run, epoch, split, ids_path, output, full_checkpoint=False):
    c=json.loads((ROOT/'configs/stage1.json').read_text())
    cmd=[sys.executable,str(ROOT/'decode_brain_adapter.py'),'--no_selection',
         '--split',split,'--image_ids',str(ids_path),'--decoded_stimuli_dir',str(output),
         '--model_weights_dir',str(run),'--saved_epochs',str(epoch),
         '--num_predictions','2','--noise_factor','4','--precision','fp32','--batch_size','1',
         '--denoising_steps','50','--seed','42','--num_workers','0']
    for k in ['data_dir','imgs_dir','parcel_dir','pretrained_model_name_or_path','subject_id','topk','condition_dim','num_decoder_queries','sub_approach']:
        cmd+=['--'+k,str(c[k])]
    if not full_checkpoint:cmd+=['--export_path',str(run/'exports'/f'epoch-{epoch}'/'inference.pt')]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)

def metrics(output, ids_path, report):
    import numpy as np
    import torch
    import metric_brain_adapter as m
    ids=np.loadtxt(ids_path,dtype=int).reshape(-1)
    payload=[np.load(output/f'image_{i:06d}.npz') for i in ids]
    for i,p in zip(ids,payload):assert int(p['image_id'])==int(i)
    originals=torch.stack([torch.from_numpy(p['groundtruth_image']) for p in payload])
    evaluator=m.MetricEvaluator(device='cuda')
    report.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for k in range(2):
        predicted=torch.stack([torch.from_numpy(p['candidate_images'][k].copy()) for p in payload]).permute(0,3,1,2).float()/255
        # Identical to the author's load_and_validate_data resize branch.
        predicted=m.resize_images(predicted,originals.shape[2:])
        with torch.no_grad():detailed,summary=evaluator.compute_all_metrics(originals,predicted)
        detailed.insert(0,'image_id',ids)
        detailed.to_csv(report/f'candidate-{k}.csv',index=False)
        summaries.append(summary['mean'].to_dict())
    mean={key:sum(s[key] for s in summaries)/2 for key in summaries[0]}
    result={'n':len(ids),'candidate_metrics':summaries,'mean':mean,
            'S':sum(mean[k] for k in ['Alex(5)','Incep','CLIP'])/3,
            'candidate_selection':False,'image_ids':ids.tolist()}
    (report/'metrics.json').write_text(json.dumps(result,indent=2))
    return result

def atlas(output, ids_path, report):
    import numpy as np
    from PIL import Image,ImageDraw
    ids=np.loadtxt(ids_path,dtype=int).reshape(-1)
    report.mkdir(parents=True,exist_ok=True)
    for part in range((len(ids)+7)//8):
        selected=ids[part*8:(part+1)*8]
        canvas=Image.new('RGB',(3*256,40+len(selected)*280),'white');draw=ImageDraw.Draw(canvas)
        for col,label in enumerate(['GT','Candidate 0','Candidate 1']):draw.text((col*256+8,10),label,fill='black')
        for row,i in enumerate(selected):
            p=np.load(output/f'image_{i:06d}.npz')
            gt=(p['groundtruth_image'].transpose(1,2,0)*255).round().astype('uint8')
            for col,img in enumerate([gt,*p['candidate_images']]):canvas.paste(Image.fromarray(img).resize((256,256)),(col*256,40+row*280))
            draw.text((8,40+row*280+258),f'image_id={i}',fill='black')
        canvas.save(report/f'atlas-{part+1}.jpg',quality=90)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--epoch',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args();run=ROOT/args.run;epoch=args.epoch
    config=ROOT/'configs';out=run/'decoded'/f'epoch-{epoch}'
    if args.smoke or epoch==1:
        decode(run,epoch,'val',config/'decode2_ids.txt',out/'export')
        decode(run,epoch,'val',config/'decode2_ids.txt',out/'checkpoint',full_checkpoint=True)
        import numpy as np
        checks=[]
        for i in np.loadtxt(config/'decode2_ids.txt',dtype=int):
            a=np.load(out/'export'/f'image_{i:06d}.npz');b=np.load(out/'checkpoint'/f'image_{i:06d}.npz')
            assert np.array_equal(a['candidate_images'],b['candidate_images'])
            checks.append({'image_id':int(i),'decoded_export_matches_checkpoint':True})
        (out/'reload_check.json').write_text(json.dumps(checks,indent=2))
        if args.smoke:
            metrics(out/'export',config/'decode2_ids.txt',ROOT/'reports/smoke-decode')
        return
    ids=config/('val_ids.txt' if epoch in (20,50,80) else 'monitor100_ids.txt')
    if not args.metrics_only:
        decode(run,epoch,'val',ids,out/'val')
        decode(run,epoch,'train',config/'atlas_train16_ids.txt',out/'train')
    report=ROOT/'reports'/f'epoch-{epoch}'
    metrics(out/'val',config/'monitor100_ids.txt',report/'val100')
    if epoch in (20,50,80):metrics(out/'val',config/'val_ids.txt',report/'val500')
    metrics(out/'train',config/'atlas_train16_ids.txt',report/'train16')
    atlas(out/'val',config/'atlas_val32_ids.txt',report/'val-atlas')
    atlas(out/'train',config/'atlas_train16_ids.txt',report/'train-atlas')
    (report/'COMPLETE').write_text('All required generation, metrics and fixed atlases finished.\n')

if __name__=='__main__':main()
