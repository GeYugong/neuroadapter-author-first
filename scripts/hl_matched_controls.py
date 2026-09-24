"""Metadata-only feasibility bounds for the frozen L120 HL matched-control plan."""
from pathlib import Path
import csv, json, hashlib, datetime, subprocess
import numpy as np
import nibabel as nib
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/hl-matched-controls-l120'
PILOT = ROOT / 'reports/ibbi-roi-l120-pilot'
RUN = ROOT / 'runs/subject01-author-fp32-seed42-lr5e-5-from100'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def components(tokens, edges):
    graph = {int(t): set() for t in tokens}
    for a, b in edges:
        if a in graph and b in graph:
            graph[a].add(b); graph[b].add(a)
    unseen = set(graph); result = []
    while unseen:
        seed = min(unseen); unseen.remove(seed); todo = [seed]; found = []
        while todo:
            v = todo.pop(); found.append(v)
            for n in sorted(graph[v] & unseen):
                unseen.remove(n); todo.append(n)
        result.append(sorted(found))
    return sorted(result, key=lambda x: (-len(x), x)), graph

def main():
    assert not (REPORT / 'feasibility.json').exists(), 'Do not overwrite an existing feasibility run'
    protocol = json.loads((REPORT / 'protocol.json').read_text())
    frozen = json.loads((PILOT / 'roi_mapping.json').read_text())
    expected = json.loads((PILOT / 'mapping-characteristics.json').read_text())
    with (PILOT / 'parcel_roi_mapping.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['token_index']) for r in rows] == list(range(200))
    hl = set(frozen['groups']['HL'])
    assert len(hl) == 78
    vertices = torch.load(RUN / 'exports/epoch-120/parcel_vertices.pt', map_location='cpu', weights_only=True)
    inputs = [PILOT / 'roi_mapping.json', PILOT / 'parcel_roi_mapping.csv',
              RUN / 'exports/epoch-120/parcel_vertices.pt', RUN / 'exports/epoch-120/metadata.json',
              RUN / 'exports/epoch-120/inference.pt', RUN / 'epochs.jsonl',
              ROOT / 'reports/best_so_far.json']
    edges = set()
    for hi, hemi in enumerate(['lh', 'rh']):
        p = ROOT / f'runtime/ibbi-roi-l120-assets/{hemi}.fsaverage.white'
        source = next(x for x in expected['spatial_assets'] if x['hemisphere'] == hemi)
        assert sha(p) == source['sha256']
        coords, faces = nib.freesurfer.read_geometry(p)
        assert coords.shape == (163842, 3)
        labels = np.full(len(coords), -1, dtype=int)
        for j, vs in enumerate(vertices[hemi]):
            inds = vs.numpy()
            token = hi * 100 + j
            assert len(inds) == int(rows[token]['valid_vertex_count'])
            assert rows[token]['hemisphere'] == hemi and len(np.unique(inds)) == len(inds)
            assert np.all(labels[inds] == -1)
            labels[inds] = token
        mesh_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
        e = labels[mesh_edges]
        e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] != e[:, 1])]
        edges.update(map(tuple, np.unique(np.sort(e, axis=1), axis=0).tolist()))
        inputs.append(p)
    assert sum(a in hl and b in hl for a, b in edges) == expected['groups_and_masks']['HL']['selected_parcel_adjacent_pairs']
    for name, tokens in frozen['mask_sets'].items():
        cs, graph = components(tokens, edges)
        prior = expected['groups_and_masks'][name]
        assert [len(c) for c in cs] == prior['connected_component_sizes']
        assert sum(map(len, graph.values())) // 2 == prior['selected_parcel_adjacent_pairs']
    locks = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert locks[str((RUN / 'exports/epoch-120/inference.pt').relative_to(ROOT))] == protocol['model_sha256']
    save(REPORT / 'input-lock.json', locks)
    with (REPORT / 'selected200-edges.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['token_a', 'token_b']); w.writerows(sorted(edges))
    hemispheres = {}; violations = []
    limits = protocol['constraints']
    for hi, hemi in enumerate(['lh', 'rh']):
        all_tokens = list(range(100 * hi, 100 * (hi + 1)))
        target = sorted(hl.intersection(all_tokens))
        pool = sorted(set(all_tokens) - hl)
        k = protocol['target_counts'][hemi]
        assert len(target) == k
        target_cs, target_g = components(target, edges)
        pool_cs, pool_g = components(pool, edges)
        required_largest_min = len(target_cs[0]) - limits['per_hemisphere_largest_component_absolute_tolerance']
        available_largest_max = len(pool_cs[0])
        if available_largest_max < required_largest_min:
            violations.append({'hemisphere': hemi, 'constraint': 'largest_connected_component',
                'required_minimum': required_largest_min, 'provable_upper_bound': available_largest_max,
                'proof': 'Every connected component of an induced subgraph is contained in one full allowed-pool component. Selecting fewer vertices cannot join disconnected components.'})
        degrees = sorted([len(v) for v in pool_g.values()], reverse=True)
        edge_upper = min(sum(degrees) // 2, sum(degrees[:k]) // 2, k * (k - 1) // 2)
        target_edge_count = sum(map(len, target_g.values())) // 2
        required_edges_min = int(np.ceil(target_edge_count * (1 - limits['per_hemisphere_internal_edges_relative_tolerance']) - 1e-12))
        if edge_upper < required_edges_min:
            violations.append({'hemisphere': hemi, 'constraint': 'internal_edges',
                'required_minimum': required_edges_min, 'provable_upper_bound': edge_upper,
                'proof': 'Selected induced edges cannot exceed either all allowed-pool edges or half the sum of its k largest full-pool degrees.'})
        cumulative = np.cumsum([len(c) for c in pool_cs])
        min_components = int(np.searchsorted(cumulative, k) + 1)
        if min_components > len(target_cs) + limits['per_hemisphere_component_count_absolute_tolerance']:
            violations.append({'hemisphere': hemi, 'constraint': 'component_count',
                'required_maximum': len(target_cs) + limits['per_hemisphere_component_count_absolute_tolerance'],
                'provable_lower_bound': min_components,
                'proof': 'At least this many full-pool components must contribute to any k selected vertices; induced selection cannot merge those components.'})
        features = {}
        for label, getter in [('parcel_snr', lambda r: float(r['snr_mean'])),
                              ('log_valid_vertices', lambda r: float(np.log(int(r['valid_vertex_count']))))]:
            values = np.array([getter(r) for r in rows])
            sd = float(values[all_tokens].std(ddof=0))
            lower = float(np.sort(values[pool])[:k].mean())
            upper = float(np.sort(values[pool])[-k:].mean())
            mean = float(values[target].mean())
            tolerance = .10 * sd
            if upper < mean - tolerance or lower > mean + tolerance:
                violations.append({'hemisphere': hemi, 'constraint': label + '_mean',
                    'required_interval': [mean - tolerance, mean + tolerance],
                    'provable_possible_interval': [lower, upper],
                    'proof': 'The average of any k selected values lies between the averages of the k smallest and k largest allowed values.'})
            features[label] = {'target_mean': mean, 'population_sd_ddof0': sd,
                'target_quantiles': np.quantile(values[target], [.25, .5, .75]).tolist(),
                'possible_mean_bounds': [lower, upper], 'required_mean_tolerance': tolerance}
        hemispheres[hemi] = {'target_tokens': target, 'target_count': k, 'allowed_pool_tokens': pool,
            'allowed_pool_count': len(pool), 'target_component_token_lists': target_cs,
            'allowed_pool_component_token_lists': pool_cs, 'target_internal_edges': target_edge_count,
            'allowed_pool_internal_edges': sum(map(len, pool_g.values())) // 2,
            'selection_internal_edges_upper_bound': edge_upper, 'selection_component_count_lower_bound': min_components,
            'target_largest_component': len(target_cs[0]), 'required_largest_component_minimum': required_largest_min,
            'selection_largest_component_upper_bound': available_largest_max, 'feature_mean_bounds': features}
    latest = json.loads((RUN / 'epochs.jsonl').read_text().splitlines()[-1])
    assert latest['epoch'] == 140 and latest['global_optimizer_updates'] == 74480
    result = {'status': 'PROVEN_INFEASIBLE' if violations else 'BOUNDS_PASSED_SEARCH_NOT_YET_RUN',
        'time': datetime.datetime.now().astimezone().isoformat(), 'hemispheres': hemispheres,
        'violations': violations, 'graph_verified_against_all_original_frozen_masks': True,
        'search_proposals_used': 0, 'control_sets_generated': 0, 'new_candidate_images': 0,
        'new_training_updates': 0, 'latest_training_progress': latest,
        'note': 'Passing bounds does not prove feasibility. Failing any rigorous bound proves no set can satisfy the full frozen constraints.'}
    save(REPORT / 'feasibility.json', result)
    unchanged = {p: sha(ROOT / p) == h for p, h in locks.items() if not p.endswith('inference.pt')}
    assert all(unchanged.values())
    save(REPORT / 'metadata-run-integrity.json', {'non_weight_inputs_rehashed_after': unchanged,
        'model_weight_sha_verified_before': protocol['model_sha256'],
        'model_weight_opened_readonly_for_sha_only': True, 'no_model_or_optimizer_constructed': True,
        'no_gpu_generation_or_training': True})
    print(json.dumps({'status': result['status'], 'violations': violations,
        'components': {h: {'HL': [len(c) for c in v['target_component_token_lists']],
                           'pool': [len(c) for c in v['allowed_pool_component_token_lists']]} for h, v in hemispheres.items()}}), flush=True)
    if violations:
        (REPORT / 'STOPPED_INFEASIBLE').write_text('Exact metadata graph bounds contradict frozen matching criteria; no search, inference or training started.\n')
    return result

if __name__ == '__main__':
    main()
