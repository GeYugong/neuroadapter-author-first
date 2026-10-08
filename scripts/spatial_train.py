"""Isolated L450 continuation with one additional image-space loss."""
import json,shutil
from pathlib import Path
import balanced_train as entry
import spatial_resume
import spatial_objective as obj
entry.balanced_resume=spatial_resume
base_install=entry.install

def install(args):
    args.spatial_objective=dict(weight=obj.LAMBDA,tmax=obj.TMAX,size=obj.SIZE)
    micro=[0]
    def observe(base,extra,n,ts):
        micro[0]+=1
        if micro[0]<=4 or micro[0]%100==0:
            row=dict(microbatch=micro[0],next_update=args.progress.global_optimizer_updates+1,rank=int(__import__('os').environ.get('RANK','0')),epsilon_loss=float(base.detach()),spatial_loss=float(extra.detach()),weighted_spatial=float(extra.detach())*obj.LAMBDA,eligible=n,timesteps=ts.detach().cpu().tolist())
            with (Path(args.output_dir)/f"objective-rank{row['rank']}.jsonl").open('a') as f:f.write(json.dumps(row)+chr(10))
    obj.install(entry.t,observer=observe)
    original_export=entry.t.export_inference
    base_install(args)
    def export(*pa,**kw):
        if pa[-1] in (475,500):return original_export(*pa,**kw)
    entry.t.export_inference=export
    def cleanup(directory,keep_last_n=2):
        directory=Path(directory).resolve()
        assert directory==entry.ROOT/'runs'/spatial_resume.BRANCH
        cps=sorted([p for p in directory.glob('checkpoint-*') if (p/'COMPLETE').exists()],key=lambda p:int(p.name.split('-')[-1]))
        for cp in cps[:-keep_last_n]:
            if int(cp.name.split('-')[-1]) not in (475,500):
                assert cp.resolve().parent==directory
                shutil.rmtree(cp)
        if shutil.disk_usage(entry.ROOT).free<25*2**30:raise RuntimeError('Complete checkpoint saved; disk reserve low')
    entry.t.cleanup_checkpoints=cleanup
entry.install=install
if __name__=='__main__':entry.main()
