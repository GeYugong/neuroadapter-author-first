"""Score the completed L450 selection without generating or selecting again."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
B = R / 'benchmarks/subj01-v1'
O = R / 'reports/l450-best8-test982'
sys.path.insert(0, str(R / 'scripts'))
import evaluate_best8 as original


def write(name, value):
    original.write(O / name, original.stamp(**value))


def preflight():
    import numpy as np
    auth = original.frozen()
    ids = np.loadtxt(original.IDS, dtype=int).reshape(-1)
    assert len(ids) == len(set(ids.tolist())) == 982
    assert len(list((O / 'records').glob('*.json'))) == 982
    assert len(list((O / 'selected').glob('*.npz'))) == 982
    assert not list((O / 'queue').glob('*.npz'))
    assert not (O / 'metrics/metrics.json').exists(), 'Metrics already exist; inspect, do not overwrite'
    all_seeds = set()
    for value in ids:
        i = int(value)
        rec = json.loads((O / 'records' / f'{i}.json').read_text())
        seeds = [original.seed_for(i, c) for c in range(8)]
        assert rec['image_id'] == i and rec['seeds'] == seeds
        assert len(set(seeds)) == len(set(rec['candidate_pixel_sha256'])) == 8
        assert all(s not in all_seeds for s in seeds)
        all_seeds.update(seeds)
        scores = np.array(rec['brain_correlations'])
        assert scores.shape == (8,) and np.isfinite(scores).all()
        idx = int(np.argmax(scores))
        assert idx == rec['selected_index'] and rec['no_gt_image_in_selection']
        p = O / 'selected' / f'image_{i:06d}.npz'
        assert original.sha(p) == rec['output_sha256']
        with np.load(p) as z:
            pixels = z['candidate_images']
            assert pixels.shape == (1, 512, 512, 3) and pixels.dtype == np.uint8
            assert int(z['image_id']) == i and int(z['selected_index']) == idx
            assert z['all_seeds'].tolist() == seeds and z['seeds'].tolist() == [seeds[idx]]
            assert np.array_equal(z['brain_correlations'], scores)
            assert original.digest(pixels[0]) == rec['selected_pixel_sha256'] == rec['candidate_pixel_sha256'][idx]
        with np.load(original.OLD / 'decoded' / p.name) as z:
            assert original.digest(z['candidate_images'][0]) == rec['candidate_pixel_sha256'][0]
    sys.path.insert(0, str(B))
    spec = importlib.util.find_spec('metric_feature_batches')
    assert Path(spec.origin).resolve() == B / 'metric_feature_batches.py'
    import metric_feature_batches
    assert callable(metric_feature_batches.install)
    # Preserve the failed evaluator and its logs. Only repair its import search path.
    source = (O / 'evaluate-isolated.py').read_text()
    target = O / 'evaluate-score-recovery2.py'
    repaired = 'import sys\nsys.path.insert(0, ' + repr(str(B)) + ')\n' + source
    compile(repaired, str(target), 'exec')
    if target.exists():
        assert target.read_text() == repaired
    else:
        with target.open('x') as f:
            f.write(repaired)
    audit = dict(n=982, K=8, unique_seeds=len(all_seeds), selection_records_verified=True,
                 candidate0_matches_K1=True, protected_files=len(auth['protected']),
                 original_wrapper_sha256=original.sha(O / 'evaluate-isolated.py'),
                 repaired_wrapper_sha256=original.sha(target),
                 fix='Prepend benchmark directory to sys.path; evaluator body unchanged',
                 metric_formula_unchanged=True, negative_pool=982, training_updates=0,
                 new_candidates=0, reselections=0)
    write('score-recovery2-preflight.json', audit)
    return audit


def free_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    occupied = {line.split(',')[0].strip() for line in jobs.splitlines() if line.strip()}
    for row in rows.splitlines():
        index, uuid, used = [s.strip() for s in row.split(',')]
        if uuid not in occupied and int(used) < 512:
            return index
    return None


def controller():
    with (O / 'score-recovery2.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (O / 'score-recovery2-started.json').exists(), 'Inspect previous attempt before any retry'
        audit = preflight()
        write('score-recovery2-started.json', dict(pid=os.getpid(), phase='waiting_for_free_gpu', preflight=audit))
        while True:
            gpu = free_gpu()
            if gpu is not None:
                break
            write('score-recovery2-status.json', dict(status='waiting_for_free_gpu', pid=os.getpid()))
            time.sleep(30)
        original.diskcheck()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        command = [str(original.PY), '-B', str(O / 'evaluate-score-recovery2.py'), '--method', 'neuroadapter', '--ids', str(original.IDS), '--feature-batch', '16']
        write('score-recovery2-status.json', dict(status='scoring', gpu=gpu, pid=os.getpid(), command=command))
        with (O / 'score-recovery2.log').open('x') as log:
            child = subprocess.Popen(command, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
            write('score-recovery2-child.json', dict(pid=child.pid, gpu=gpu, command=command))
            code = child.wait()
        write('score-recovery2-exit.json', dict(returncode=code))
        assert code == 0, 'Scoring failed; inspect log and preserve all outputs'
        original.frozen()
        import pandas as pd
        import numpy as np
        new = json.loads((O / 'metrics/metrics.json').read_text())
        old = json.loads((original.OLD / 'metrics/metrics.json').read_text())
        frame = pd.read_csv(O / 'metrics/per-image.csv')
        assert frame['nsd_id0'].tolist() == np.loadtxt(original.IDS, dtype=int).tolist()
        assert new['n_eval'] == 982 and new['K'] == 8
        for k, v in new['mean'].items():
            assert np.isfinite(v) and np.isclose(frame[k].mean(), v, rtol=0, atol=1e-10)
        write('comparison.json', dict(K1=old['mean'], best8=new['mean'], delta={k:new['mean'][k]-old['mean'][k] for k in old['mean']}, same_weight=True, candidate0_matches_K1=True, training_updates=0, atlas_review_pending=True, limitations=auth_limitations()))
        write('score-recovery2-status.json', dict(status='metrics_complete', n=982, K=8, gpu=gpu, means_verified=True, deliverable_complete=False))


def auth_limitations():
    return json.loads((O / 'authorization.json').read_text())['limitations']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'controller'])
    args = parser.parse_args()
    try:
        if args.mode == 'preflight':
            print(json.dumps(preflight()), flush=True)
        else:
            controller()
    except BaseException:
        write('score-recovery2-failure.json', dict(traceback=traceback.format_exc()))
        raise
