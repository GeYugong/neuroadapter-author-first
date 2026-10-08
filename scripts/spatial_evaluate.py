"""Reuse frozen K2 metrics, with isolated branch paths."""
import argparse
import balanced_evaluate as entry
entry.RUN=entry.ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-spatiall1-from450'
entry.REPORT=entry.ROOT/'reports/spatial-l1-from450'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,choices=[475,500],required=True);p.add_argument('--phase',choices=['decode','score'],required=True);a=p.parse_args()
    progress=entry.read(entry.RUN/f'checkpoint-{a.epoch}/progress.json')
    assert progress['completed_epochs']==a.epoch and progress['skipped_updates']==0
    entry.phase(a.epoch,a.phase)
