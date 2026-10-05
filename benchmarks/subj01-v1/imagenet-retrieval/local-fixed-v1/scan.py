"""Complete exact ImageNet scan, measured fMRI queries only, resumable blocks."""
import argparse
import concurrent.futures
import fcntl
import io
import os
import sys
sys.dont_write_bytecode = True
from common import *
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
import torch
from torch.utils.data import DataLoader
from corpus import Candidates, prepare_manifest
from encoder import HalfEncoder


def atomic_state(path, **arrays):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Pearson:
    def __init__(self, queries):
        q = torch.from_numpy(queries.astype(np.float64)).to('cuda:0')
        q -= q.mean(dim=1, keepdim=True)
        norms = q.norm(dim=1, keepdim=True)
        assert torch.isfinite(q).all() and (norms > 0).all()
        self.q = (q / norms).T.contiguous()

    @torch.inference_mode()
    def scores(self, predictions):
        x = torch.from_numpy(predictions.astype(np.float64)).to('cuda:0')
        x -= x.mean(dim=1, keepdim=True)
        norms = x.norm(dim=1, keepdim=True)
        assert torch.isfinite(x).all() and (norms > 0).all(), 'Invalid candidate spatial norm'
        return ((x / norms) @ self.q).cpu().numpy()


def update(best, winners, scores, indices):
    indices = np.asarray(indices)
    # Explicit candidate-ID order handles ties even with reversed input blocks.
    order = np.argsort(indices)
    s, indices = scores[order], indices[order]
    local = s.argmax(axis=0)
    values, chosen = s[local, np.arange(s.shape[1])], indices[local]
    take = (values > best) | ((values == best) & (chosen < winners))
    best[take], winners[take] = values[take], chosen[take]


def progress(done, started_at, run_start, first, binding, last_checkpoint):
    elapsed = time.monotonic() - started_at
    rate = (done - first) / elapsed if elapsed > 0 else 0
    d = dict(status='scanning', execution_complete=False, scanned_candidates=done,
        total_candidates=1281167, run_started_utc=run_start, updated_utc=utc(),
        current_run_elapsed_seconds=elapsed, candidates_per_second=rate,
        eta_hours=(1281167-done)/rate/3600 if rate else None,
        binding=binding, latest_checkpoint=str(last_checkpoint), **limitations())
    write_json(HERE / 'run-status.json', d)
    print(json.dumps(d, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    a = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    lock = (ASSETS / 'scan.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (HERE / 'query-audit.json').exists(), 'Queries must pass before retrieval'
    assert not (HERE / 'retrieval-complete.json').exists(), 'Completed scan must not be repeated'
    qa = json.loads((HERE / 'query-audit.json').read_text())
    assert qa['status'] == 'passed' and sha(ASSETS / 'queries.npy') == qa['query_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    prepare_manifest()
    corpus = json.loads((HERE / 'corpus.json').read_text())
    audit = dict(started_utc=utc(), **limitations(), nsd_gt_images_loaded=0)
    con = __import__('sqlite3').connect(f'file:{ASSETS / "candidates.sqlite"}?mode=ro', uri=True)
    gray = con.execute('SELECT idx FROM candidates WHERE id=?', ('n01440764_15560.JPEG',)).fetchone()[0]
    sample_ids = sorted(set(list(range(31)) + [gray]))
    # Fixed candidates independent of targets and scores; includes a gray image.
    dataset = Candidates(sample_ids)
    records = [dataset[i] for i in range(len(dataset))]
    sample = torch.stack([r[0] for r in records])
    halves = [HalfEncoder('lh', 'cuda:0', sample), HalfEncoder('rh', 'cuda:1', sample)]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    def predict(batch, use_cache=True):
        jobs = [pool.submit(h.predict, batch, use_cache) for h in halves]
        return np.concatenate([j.result() for j in jobs], axis=1)
    batch_size = 16
    first = predict(sample[:2])
    uncached = predict(sample[:2], False)
    assert np.allclose(first, uncached, atol=2e-4, rtol=2e-4)
    # The actual production batch shape, including numerical comparison to B2.
    production = predict(sample[:batch_size])
    assert np.allclose(first, production[:2], atol=2e-4, rtol=2e-4)
    predictions = np.concatenate([production, predict(sample[batch_size:])])
    queries = np.load(ASSETS / 'queries.npy')
    scoring = Pearson(queries)
    scores = scoring.scores(predictions)
    ref = np.corrcoef(np.concatenate([predictions.astype(np.float64), queries.astype(np.float64)]))[:len(predictions), len(predictions):]
    max_error = float(np.abs(scores - ref).max())
    assert max_error < 2e-12, max_error
    direct, direct_ids = np.full(982, -np.inf), np.full(982, 2**63-1, dtype=np.int64)
    update(direct, direct_ids, ref, sample_ids)
    saved = ASSETS / 'preflight-resume.npz'
    best, win = np.full(982, -np.inf), np.full(982, 2**63-1, dtype=np.int64)
    update(best, win, scores[:13], sample_ids[:13])
    atomic_state(saved, best=best, winners=win, next_index=13)
    with np.load(saved) as s:
        best, win = s['best'].copy(), s['winners'].copy()
    for start in reversed(range(13, len(sample_ids), 3)):
        update(best, win, scores[start:start+3], sample_ids[start:start+3])
    assert np.array_equal(win, direct_ids) and np.allclose(best, direct, rtol=0, atol=2e-12)
    # Exact ties: deterministic lower original candidate ID wins, no epsilon tie.
    tb, tw = np.array([-np.inf]), np.array([2**63-1], dtype=np.int64)
    update(tb, tw, np.array([[.5], [.5]]), [10, 3]); assert tw[0] == 3
    update(tb, tw, np.array([[.5]]), [1]); assert tw[0] == 1
    sorted_scores = np.sort(scores, axis=0)
    margin = sorted_scores[-1] - sorted_scores[-2]
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    t = time.monotonic()
    for _ in range(4):
        scoring.scores(predict(sample[:batch_size]))
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    throughput = 4 * batch_size / (time.monotonic() - t)
    audit.update(status='passed', member_checks=[r for h in halves for r in h.audit],
        dino_shared_vs_uncached_max_abs=float(np.abs(first-uncached).max()),
        production_batch_vs_batch2_max_abs=float(np.abs(first-production[:2]).max()),
        stream_vs_float64_reference_max_abs=max_error, interrupted_resumed_reversed_blocks_top1_equal=True,
        exact_tie_lower_candidate_id_test=True, fixed_candidate_indices=sample_ids,
        minimum_top1_margin=float(margin.min()), near_ties_within_2e12=int((margin <= 2e-12).sum()),
        batch_size=batch_size, measured_encoder_and_scoring_candidates_per_second=throughput,
        gpu_peak_bytes=[torch.cuda.max_memory_allocated(i) for i in [0, 1]], finished_utc=utc())
    if not (HERE / 'preflight.json').exists():
        write_json(HERE / 'preflight.json', audit)
    else:
        write_json(HERE / ('preflight-resume-' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '.json'), audit)
    print('PREFLIGHT PASSED', json.dumps(audit), flush=True)
    protocol = dict(version='local-fixed-v1', **limitations(), subject=1,
        n_queries=982, eval_ids_sha256=sha(BENCH / 'test_full_ids.txt'),
        query_sha256=qa['query_sha256'], query_manifest_sha256=qa['query_manifest_sha256'],
        query_formula=qa['formula'], session_statistics_include_evaluation_trials=True,
        mask_sha256=sha(MASK), mask_vertices=[16236, 17380], streams=False,
        mask_order='LH ascending fsaverage vertex ID then RH ascending ID',
        official_asset_verification_sha256=sha(HERE.parent / 'official-weight-verification.json'),
        existing_encoder_audit_sha256=sha(HERE.parent / 'public-asset-audit.json'),
        corpus_manifest_sha256=corpus['manifest_sha256'], n_candidates=1281167,
        ensemble='16 members; softmax(20*val_corr) across 8 members per hemisphere per vertex',
        inference_dtype='float32', tf32=False, production_batch_size=batch_size,
        correlation='float64 Pearson once across concatenated33616 vertices, all982queries for every candidate',
        tie_rule='exact float64 equality chooses lexicographically smaller original candidate filename',
        preprocessing='RGB; ToTensor; Resize425 bilinear antialias=True; CenterCrop425; ImageNet Normalize; right/bottom normalized-zero pad434',
        retrieved_view='425x425 float32 tensor before Normalize; no padding; PNG preview only',
        checkpoint_interval_candidates=1024,
        code_sha256={n: sha(HERE / n) for n in ['common.py', 'prepare_queries.py', 'encoder.py', 'corpus.py', 'scan.py']})
    if (HERE / 'protocol.json').exists():
        assert json.loads((HERE / 'protocol.json').read_text()) == protocol, 'Protocol changed; refuse mixed resume'
    else:
        write_json(HERE / 'protocol.json', protocol)
    binding = sha(HERE / 'protocol.json')
    if a.preflight_only:
        return
    checkpoints = ASSETS / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    journal = ASSETS / 'candidate-audit'; journal.mkdir(exist_ok=True)
    existing = sorted(checkpoints.glob('state-*.npz'))
    best, win, start = np.full(982, -np.inf), np.full(982, 2**63-1, dtype=np.int64), 0
    if existing:
        assert a.resume, 'Existing scan requires explicit --resume'
        with np.load(existing[-1]) as s:
            assert s['binding'].item() == binding
            best, win, start = s['best'].copy(), s['winners'].copy(), int(s['next_index'])
        # Every committed prefix must have its durable per-file audit journal.
        assert sum(1 for p in journal.glob('block-*.jsonl') for _ in p.open()) >= start
    else:
        assert not a.resume, 'No checkpoint to resume'
    loader = DataLoader(Candidates(range(start, 1281167)), batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, multiprocessing_context='spawn', prefetch_factor=2)
    run_start, t0 = utc(), time.monotonic()
    scanned, block_start, block_records = start, start, []
    for batch, indices, audit_rows in loader:
        indices = indices.numpy()
        assert np.array_equal(indices, np.arange(scanned, scanned+len(batch)))
        values = scoring.scores(predict(batch))
        update(best, win, values, indices)
        block_records.extend(audit_rows)
        scanned += len(batch)
        if scanned % 1024 == 0 or scanned == 1281167:
            jp = journal / f'block-{block_start:07d}-{scanned:07d}.jsonl'
            tmp = jp.with_suffix('.tmp')
            with tmp.open('w') as f:
                f.write('\n'.join(block_records) + '\n'); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, jp)
            cp = checkpoints / f'state-{scanned:07d}.npz'
            atomic_state(cp, best=best, winners=win, next_index=scanned, binding=binding,
                         last_journal_sha256=sha(jp))
            progress(scanned, t0, run_start, start, binding, cp)
            block_records, block_start = [], scanned
    assert scanned == 1281167 and np.isfinite(best).all()
    results = []
    ids = np.loadtxt(BENCH / 'test_full_ids.txt', dtype=np.int64)
    for i, winner, r in zip(ids, win, best):
        cid, path, size = con.execute('SELECT id,path,size FROM candidates WHERE idx=?', (int(winner),)).fetchone()
        results.append(dict(nsd_id0=int(i), candidate_index0=int(winner), candidate_id=cid,
                            candidate_path=path, pearson=float(r), source_sha256=sha(path)))
    write_json(HERE / 'top1.json', dict(protocol_sha256=binding, records=results, **limitations()))
    record_count = 0
    audit_hash = hashlib.sha256()
    for p in sorted(journal.glob('block-*.jsonl')):
        for line in p.open('rb'):
            row = json.loads(line)
            assert row['idx'] == record_count, (p, record_count, row['idx'])
            record_count += 1; audit_hash.update(line)
    assert record_count == 1281167
    complete = dict(retrieval_complete=True, execution_complete=False, scanned_candidates=scanned,
        source_hash_and_decode_records=record_count, concatenated_candidate_audit_sha256=audit_hash.hexdigest(),
        protocol_sha256=binding, top1_sha256=sha(HERE / 'top1.json'), finished_utc=utc(),
        remaining='Independent metrics, comparison tables, plots and atlas review', **limitations())
    write_json(HERE / 'retrieval-complete.json', complete)
    write_json(HERE / 'run-status.json', dict(status='retrieval_complete_evaluation_pending', **complete))
    pool.shutdown()
    print('FULL RETRIEVAL COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        write_json(HERE / ('fault-' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '.json'),
                   dict(utc=utc(), error=repr(e), traceback=traceback.format_exc(), **limitations()))
        raise
