"""Prepare measured-only queries; no stimulus pixels or image features loaded."""
import csv
import hashlib
import json
import os
import sys
sys.dont_write_bytecode = True
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from common import *


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    if (HERE / 'query-audit.json').exists():
        a = json.loads((HERE / 'query-audit.json').read_text())
        assert sha(ASSETS / 'queries.npy') == a['query_sha256']
        print('Existing verified queries retained', flush=True)
        return
    ids_file = BENCH / 'test_full_ids.txt'
    assert sha(ids_file) == '4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441'
    ids = np.loadtxt(ids_file, dtype=np.int64)
    assert len(ids) == len(set(ids)) == 982
    design_file = RAW / 'experiments/nsd/nsd_expdesign.mat'
    design = loadmat(design_file)
    subject = design['subjectim'][0].astype(np.int64) - 1
    order = design['masterordering'].reshape(-1).astype(np.int64) - 1
    trial_ids = subject[order]
    metadata = np.load(ROOT / 'data/metadata_sub-01.npy', allow_pickle=True).item()
    assert np.array_equal(trial_ids, metadata['img_presentation_order'])
    assert len(trial_ids) == 30000 and all(np.count_nonzero(trial_ids == i) == 3 for i in ids)
    masks = dict(np.load(MASK))
    assert [int(masks[h].sum()) for h in ['lh', 'rh']] == [16236, 17380]
    id_to_row = {int(v): k for k, v in enumerate(ids)}
    fixed5 = np.loadtxt(BENCH / 'identity5_ids.txt', dtype=np.int64).tolist()
    rows = []
    for i in ids:
        tr = np.flatnonzero(trial_ids == i)
        rows.append(dict(nsd_id0=int(i), repeat_count=len(tr), trial_indices0=tr.tolist(),
                         sessions1=(tr // 750 + 1).tolist(), session_trials0=(tr % 750).tolist()))
    audit = dict(started_utc=utc(), **limitations(), nsd_gt_images_loaded=0,
                 eval_ids_sha256=sha(ids_file), design_sha256=sha(design_file),
                 metadata_order_exact=True, fixed_five_ids=fixed5, session_records=[],
                 nonfinite_policy='Report all raw anomalies; fail if any selected vertex is nonfinite. No imputation.',
                 formula='float32 x=raw/300; per-session vertex mean/std over all750 trials ddof0; (x-mean)/(std+1e-6); chronological repeat arithmetic mean',
                 normalization_includes_evaluation_trials=True)
    queries = np.zeros((982, 33616), dtype=np.float32)
    checks = {int(i): {} for i in fixed5}
    offset = 0
    for h in ['lh', 'rh']:
        selected = np.flatnonzero(masks[h])
        sums = np.zeros((982, len(selected)), dtype=np.float32)
        counts = np.zeros(982, dtype=np.int64)
        for session in range(1, 41):
            p = RAW / f'betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/{h}.betas_session{session:02}.mgh'
            raw = nib.load(p).get_fdata(dtype=np.float32)[:, 0, 0, :].T
            assert raw.shape == (750, 163842)
            invalid = ~np.isfinite(raw)
            bad_vertices = np.flatnonzero(invalid.any(axis=0))
            bad_selected = np.intersect1d(bad_vertices, selected)
            rec = dict(hemisphere=h, session1=session, path=str(p), sha256=sha(p),
                       shape=list(raw.shape), nonfinite_count=int(invalid.sum()),
                       nonfinite_vertices0=bad_vertices.tolist(), selected_nonfinite_vertices0=bad_selected.tolist())
            if bad_selected.size:
                audit['session_records'].append(rec)
                write_json(HERE / 'query-failure.json', audit)
                raise ValueError(f'Nonfinite measured query vertices: {rec}')
            # Standardization is independent per vertex: slicing the fixed mask
            # first is mathematically identical and avoids storing unused betas.
            x = np.ascontiguousarray(raw[:, selected]) / np.float32(300)
            mean = x.mean(axis=0, dtype=np.float32)
            std = x.std(axis=0, ddof=0, dtype=np.float32)
            z = (x - mean) / (std + np.float32(1e-6))
            assert np.isfinite(z).all()
            rec['constant_selected_vertices'] = int(np.count_nonzero(std == 0))
            stats = ASSETS / f'statistics-{h}-{session:02}.npz'
            np.savez(stats, vertex_id0=selected, mean=mean, std=std)
            rec['statistics_sha256'] = sha(stats)
            local_ids = trial_ids[(session-1)*750:session*750]
            for trial, i in enumerate(local_ids):
                row = id_to_row.get(int(i))
                if row is None:
                    continue
                sums[row] += z[trial]; counts[row] += 1
                if int(i) in checks:
                    # Independent scalar checks at first/middle/last masked vertex.
                    for col in [0, len(selected)//2, len(selected)-1]:
                        vertex = int(selected[col]); scalar = raw[:, vertex].astype(np.float32) / np.float32(300)
                        expected = (scalar[trial]-scalar.mean())/(scalar.std(ddof=0)+np.float32(1e-6))
                        assert np.isclose(expected, z[trial, col], rtol=2e-5, atol=2e-5)
                        checks[int(i)].setdefault(h, []).append(dict(session1=session, trial0=trial,
                            global_trial0=(session-1)*750+trial, vertex_id0=vertex,
                            scalar_z=float(expected), block_z=float(z[trial, col])))
            audit['session_records'].append(rec)
            write_json(HERE / 'query-progress.json', dict(last_session=rec, completed=len(audit['session_records']), **limitations()))
            print(h, session, 'raw nonfinite', rec['nonfinite_count'], 'selected', len(bad_selected), flush=True)
            del raw, invalid, x, z
        assert np.all(counts == 3)
        queries[:, offset:offset+len(selected)] = sums / counts[:, None]
        offset += len(selected)
    assert np.isfinite(queries).all()
    centered = queries.astype(np.float64) - queries.astype(np.float64).mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    assert np.all(norms > 0)
    tmp = ASSETS / 'queries.tmp.npy'; np.save(tmp, queries); os.replace(tmp, ASSETS / 'queries.npy')
    write_json(HERE / 'query-manifest.json', dict(queries=rows, **limitations()))
    audit.update(query_sha256=sha(ASSETS / 'queries.npy'), query_shape=list(queries.shape),
                 fixed_five_checks=checks, minimum_spatial_norm=float(norms.min()),
                 finished_utc=utc(), status='passed', all_982_repeat_counts=3,
                 query_manifest_sha256=sha(HERE / 'query-manifest.json'))
    write_json(HERE / 'query-audit.json', audit)
    print('QUERY CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
