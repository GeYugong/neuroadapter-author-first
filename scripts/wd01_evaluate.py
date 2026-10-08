"""Reuse frozen K2 decoding and evaluation in isolated paths."""
import argparse
import balanced_evaluate as entry
entry.RUN=entry.ROOT/'runs/subject01-author-fp32-seed42-lr1e-5-wd01-from450'
entry.REPORT=entry.ROOT/'reports/wd01-from450'
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['decode','score'],required=True)
    a=ap.parse_args()
    p=entry.read(entry.RUN/'checkpoint-500/progress.json')
    assert p['completed_epochs']==500 and p['skipped_updates']==0
    entry.phase(500,a.phase)
