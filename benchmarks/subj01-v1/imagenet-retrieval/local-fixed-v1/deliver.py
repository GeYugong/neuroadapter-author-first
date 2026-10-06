"""Independent post-scan evaluation, comparisons and fixed atlas; never trains."""
import os
import sys
sys.dont_write_bytecode = True
from common import *
os.environ['TORCH_HOME'] = str(ROOT / 'runtime/torch')
os.environ['AUTHOR_SWAV_SOURCE'] = str(ROOT / 'runtime/swav')
os.environ['AUTHOR_CLIP_PATH'] = str(OLD / 'models/evaluation/clip/ViT-L-14.pt')
os.environ['MPLCONFIGDIR'] = str(ASSETS / 'mpl')
os.environ['OMP_NUM_THREADS'] = '4'
import csv
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))
import metric_brain_adapter as metric
from metric_feature_batches import install

KEYS = ['PixCorr', 'SSIM', 'Alex(2)', 'Alex(5)', 'Incep', 'CLIP', 'Eff', 'SwAV']
LABELS = ['PixCorr', 'SSIM', 'AN2', 'AN5', 'Incep', 'CLIP', 'Eff', 'SwAV']
METRIC_SHA = '646ee1f3673b033cd6ee485058db16a3d24fae7624d793fe5174008ad2e15f7d'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def comparison_sources(ids, metric_sha):
    sources = [('NeuroAdapter L400', ROOT / 'reports/lr-half-to500/test982/metrics/metrics.json')]
    sources += [(label, BENCH / f'metrics/{method}/test_full_ids/metrics.json') for label, method in
                [('MindEye1 K1', 'mindeye1'), ('Brain-Diffuser K1', 'brain-diffuser'),
                 ('Takagi K1', 'takagi'), ('MindBridge(s) K1', 'mindbridge')]]
    verified, values = [], {}
    for label, path in sources:
        if not path.exists():
            values[label] = {k: None for k in KEYS}
            verified.append(dict(method=label, status='NA', reason='Existing metrics absent'))
            continue
        data = json.loads(path.read_text())
        assert data['n_eval'] == 982 and data['K'] == 1 and data['selection'] == 'none'
        assert data['eval_ids_hash'] == sha(BENCH / 'test_full_ids.txt')
        assert data['metric_sha256'] == metric_sha
        per = pd.read_csv(path.parent / 'per-image.csv')
        assert np.array_equal(per['nsd_id0'].to_numpy(), ids)
        assert np.isfinite(per[KEYS].to_numpy()).all()
        for k in KEYS:
            assert np.isclose(per[k].mean(), data['mean'][k], rtol=1e-12, atol=1e-12)
        values[label] = data['mean']
        verified.append(dict(method=label, status='same982_and_frozen_metric_verified',
            metrics_path=str(path), metrics_sha256=sha(path),
            per_image_path=str(path.parent / 'per-image.csv'), per_image_sha256=sha(path.parent / 'per-image.csv'),
            original_record=data))
    # The L400 export is tied to the previously verified generation file hashes;
    # it is never taken from the benchmark's L120 output directory.
    audit_path = ROOT / 'reports/lr-half-to500/test982/generation-audit.json'
    generated = json.loads(audit_path.read_text())
    for i in ids:
        p = ROOT / f'reports/lr-half-to500/test982/decoded/image_{i:06d}.npz'
        assert sha(p) == generated[p.name]
    write_json(HERE / 'comparison-source-audit.json', dict(methods=verified,
        l400_generation_audit_sha256=sha(audit_path), l400_all982_output_hashes_matched=True,
        old_benchmark_neuroadapter_is_L120_not_used=True, **limitations()))
    return values


def tables_and_plot(baseline, values):
    all_values = {'ImageNet retrieval local-fixed-v1': baseline, **values}
    raw, rel, delta = [], [], []
    for method, means in all_values.items():
        r, d, c = dict(method=method), dict(method=method), dict(method=method)
        for k, name in zip(KEYS, LABELS):
            b, v = baseline[k], means.get(k)
            r[name] = v
            d[name] = None if v is None else v-b
            c[name] = None if v is None or b <= 0 else 100*((b-v) if k in ['Eff', 'SwAV'] else (v-b))/b
        for x in [r, d, c]:
            x.update(disclosure=DISCLOSURE, n_eval=982, encoder_development_overlap='unknown', result_scope='local_fixed_exploratory')
        raw.append(r); rel.append(c); delta.append(d)
    for filename, data in [('raw-metrics.csv', raw), ('relative-change-percent.csv', rel), ('absolute-difference.csv', delta)]:
        pd.DataFrame(data).to_csv(HERE / filename, index=False, na_rep='NA')
    write_json(HERE / 'tables-metadata.json', dict(**limitations(), n_eval=982,
        identification_unit='AN2/AN5/Incep/CLIP are percentages',
        higher_is_better=LABELS[:6], lower_is_better=LABELS[6:],
        relative_formula_first_six='100*(method-baseline)/baseline',
        relative_formula_eff_swav='100*(baseline-method)/baseline',
        nonpositive_denominator='NA; retain raw method-baseline difference',
        relative_values='descriptive, not leakage-verified generalization or exact Figure5 reproduction',
        excluded_protocols=['100k 50 images best-of8', 'MindEye best-of16', 'paper cited values']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=FONT)
    colors = ['#2070b4', '#dc8048', '#48977a', '#8d6bab', '#c35e70']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, (ax, name) in enumerate(zip(axes.flat, LABELS)):
        vals = [r[name] for r in rel[1:]]
        finite = [(j, v) for j, v in enumerate(vals) if v is not None]
        ax.bar([j for j, v in finite], [v for j, v in finite], color=[colors[j] for j, v in finite], width=.68)
        for j, v in enumerate(vals):
            if v is None:
                ax.text(j, 0, 'NA', ha='center', va='bottom')
        ax.axhline(0, color='#454545', linewidth=.8)
        ax.set_title(name + (' ↓ distance' if col >= 6 else ' ↑'))
        ax.set_xticks(range(5), ['L400', 'ME1', 'BD', 'Takagi', 'MB(s)'], rotation=25)
        ax.set_ylabel('Relative change (%)')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Subject1 · 同982图 · 相对实测ImageNet检索baseline的描述性变化', fontproperties=font, fontsize=18)
    fig.text(.5, .042, DISCLOSURE, ha='center', fontproperties=font, fontsize=11)
    fig.text(.5, .016, '负值保留；并非已确认无泄漏的泛化优势；仅Subject1。', ha='center', fontproperties=font, fontsize=11)
    fig.tight_layout(rect=[0, .08, 1, .95])
    fig.savefig(HERE / 'relative-change.png', dpi=180)
    fig.savefig(HERE / 'relative-change.pdf')
    plt.close(fig)


def atlas(top1, image_dir):
    ids = np.loadtxt(BENCH / 'atlas32_ids.txt', dtype=np.int64)
    assert len(ids) == 32 and set(ids) <= set(top1)
    out = HERE / 'atlas32'; out.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT, 19); small = ImageFont.truetype(FONT, 14)
    pages = []
    for page in range(4):
        canvas = Image.new('RGB', (1020, 8*352+120), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), 'Subject1 固定32目标 · GT / ImageNet检索 / NeuroAdapter L400', font=font, fill='black')
        for j, title in enumerate(['GT', 'ImageNet retrieval', 'NeuroAdapter L400']):
            draw.text((j*340+10, 38), title, font=font, fill='black')
        for row, i in enumerate(ids[page*8:(page+1)*8]):
            i = int(i); y = 72 + row*352
            gt = Image.open(BENCH / f'assets/gt/{i}.png').convert('RGB')
            retrieved = Image.open(image_dir / f'{i}.png').convert('RGB')
            with np.load(ROOT / f'reports/lr-half-to500/test982/decoded/image_{i:06d}.npz') as f:
                assert int(f['image_id']) == i and f['candidate_images'].shape[0] == 1
                l400 = Image.fromarray(f['candidate_images'][0])
            for col, im in enumerate([gt, retrieved, l400]):
                canvas.paste(im.resize((330, 330)), (col*340+5, y))
            record = top1[i]
            draw.text((8, y+332), f'NSD ID0={i} | {record["candidate_id"]} | Pearson={record["pearson"]:.6f}', font=small, fill='black')
        draw.text((8, canvas.height-35), DISCLOSURE, font=small, fill='black')
        p = out / f'page-{page+1:02}.jpg'; canvas.save(p, quality=95)
        pages.append(dict(path=str(p), sha256=sha(p), nsd_ids0=ids[page*8:(page+1)*8].tolist()))
    write_json(out / 'manifest.json', dict(pages=pages, fixed_ids_source=str(BENCH/'atlas32_ids.txt'),
        ids_sha256=sha(BENCH/'atlas32_ids.txt'), reviewer_status='pending_actual_AI_nonblind_visual_review', **limitations()))


def main():
    # Refuse to overwrite historical tensors/tables before checking old caches.
    existing_metrics = HERE / 'baseline-metrics/metrics.json'
    if existing_metrics.exists():
        existing = json.loads(existing_metrics.read_text())
        if existing.get('input_range_policy') != 'known_unit_float32_validate_epsilon_then_clamp':
            raise RuntimeError(
                'Legacy scoring artifacts are preserved; use the separate scoring correction directory.')
    complete = json.loads((HERE / 'retrieval-complete.json').read_text())
    assert complete['retrieval_complete'] and complete['scanned_candidates'] == 1281167
    assert sha(HERE / 'protocol.json') == complete['protocol_sha256']
    assert sha(HERE / 'top1.json') == complete['top1_sha256']
    assert sha(ROOT / 'metric_brain_adapter.py') == METRIC_SHA
    torch.set_num_threads(4)
    ids = np.loadtxt(BENCH / 'test_full_ids.txt', dtype=np.int64)
    records = json.loads((HERE / 'top1.json').read_text())['records']
    assert np.array_equal([r['nsd_id0'] for r in records], ids)
    top1 = {r['nsd_id0']: r for r in records}
    winning_indices = {r['candidate_index0'] for r in records}
    scanned_hashes = {}
    for journal in sorted((ASSETS / 'candidate-audit').glob('block-*.jsonl')):
        for line in journal.open():
            row = json.loads(line)
            if row['idx'] in winning_indices:
                scanned_hashes[row['idx']] = row['source_sha256']
    assert set(scanned_hashes) == winning_indices
    for r in records:
        assert r['source_sha256'] == scanned_hashes[r['candidate_index0']], 'Winning source changed since encoding'
    previews = ASSETS / 'retrieved-views'; previews.mkdir(exist_ok=True)
    views = ASSETS / 'retrieved-view-tensors'; views.mkdir(exist_ok=True)
    crop = transforms.Compose([transforms.ToTensor(), transforms.Resize(425), transforms.CenterCrop(425)])
    imgs = []
    result_rows = []
    gt_manifest = pd.read_csv(BENCH / 'test_manifest.csv').set_index('nsd_id0')
    ground = []
    for i, record in zip(ids, records):
        source = Path(record['candidate_path'])
        assert sha(source) == record['source_sha256']
        with Image.open(source) as im:
            x = crop(im.convert('RGB'))
        assert x.shape == (3, 425, 425) and torch.isfinite(x).all()
        # Tensor resize can exceed 1 by a few float32 ULPs. The frozen
        # identification metric otherwise mistakes the entire pool for uint8.
        if float(x.min()) < -1e-6 or float(x.max()) > 1.0 + 1e-6:
            raise ValueError(f"Unexpected retrieval image range for {i}")
        x = x.clamp(0, 1)
        view = views / f'{i}.npy'; np.save(view, x.numpy())
        png = previews / f'{i}.png'
        Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)).save(png)
        gt = np.array(Image.open(BENCH / f'assets/gt/{i}.png').convert('RGB'))
        assert hashlib.sha256(gt.tobytes()).hexdigest() == gt_manifest.loc[i, 'gt_sha256']
        ground.append(torch.from_numpy(gt).permute(2, 0, 1).float()/255)
        imgs.append(x)
        result_rows.append(dict(**record, retrieved_view_path=str(view), retrieved_view_sha256=sha(view),
            preview_png_path=str(png), preview_png_sha256=sha(png), disclosure=DISCLOSURE))
    pd.DataFrame(result_rows).to_csv(HERE / 'top1.csv', index=False)
    metadata = dict(n_eval=982, K=1, selection='full-corpus measured-fMRI Pearson top1',
        eval_ids_hash=sha(BENCH / 'test_full_ids.txt'), metric_sha256=METRIC_SHA,
        input_range_policy='known_unit_float32_validate_epsilon_then_clamp',
        feature_batch=16, negative_pool='all982, extract features in batches then evaluate whole pool',
        prediction_serialization='float32 encoder425x425 field of view before Normalize; PNG preview not used for scores',
        GT='original NSD425x425 RGB uint8/255', **limitations())
    output = HERE / 'baseline-metrics'; output.mkdir(exist_ok=True)
    if (output / 'metrics.json').exists():
        existing = json.loads((output / 'metrics.json').read_text())
        for k, v in metadata.items():
            assert existing[k] == v
        baseline = existing['mean']
    else:
        gt, pred = torch.stack(ground), torch.stack(imgs)
        install(metric, 16)
        with torch.inference_mode():
            detailed, summary = metric.MetricEvaluator(device='cuda:0').compute_all_metrics(gt, pred)
        assert len(detailed) == 982 and np.isfinite(detailed[KEYS].to_numpy()).all()
        detailed.insert(0, 'nsd_id0', ids)
        detailed.to_csv(output / 'per-image.csv', index=False)
        baseline = summary['mean'].to_dict()
        write_json(output / 'metrics.json', dict(**metadata, mean=baseline, finished_utc=utc()))
    values = comparison_sources(ids, METRIC_SHA)
    tables_and_plot(baseline, values)
    atlas(top1, previews)
    report = ['# 本地固定协议 v1：完整检索与评价', '', DISCLOSURE, '',
        '完整1281167候选均已扫描；982查询均来自实测fMRI，新增训练更新0。正式评分在全库top1锁定后独立执行。', '',
        '## 八项指标', '', '| 方法 | ' + ' | '.join(LABELS) + ' |',
        '|---|' + '|'.join(['---:']*8) + '|']
    for name, means in {'ImageNet local-fixed-v1': baseline, **values}.items():
        report.append('| '+name+' | '+' | '.join('NA' if means.get(k) is None else f'{means[k]:.6f}' for k in KEYS)+' |')
    report += ['', 'AN2/AN5/Incep/CLIP单位为%；Eff与SwAV为相关距离，越低越好。完整未舍入值及负相对变化见CSV。', '',
        '固定同982评价池及原指标源码，全池负例未按批拆分。L400来自lr-half-to500/test982，未混用旧benchmark中的L120。', '',
        '相对变化只是本地描述性比较，不是已验证无泄漏的泛化优势。session标准化使用当session全部trial，包含评价trial；仅Subject1。未联系作者、未修改旧模型/结果/best。', '',
        '右半球session11原始10个顶点共7500个NaN全部位于固定mask外；已记录而未填零，参与检索的全部查询均有限。', '',
        '图册页面已生成；实际AI非盲目视审阅、三端发布核验完成前，execution_complete仍为false。']
    (HERE / 'RESULTS_ZH.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    write_json(HERE / 'delivery-artifacts.json', dict(status='metrics_tables_plots_and_atlas_generated',
        execution_complete=False, pending=['actual_AI_nonblind_atlas_review', 'server_local_private_GitHub_sync'],
        baseline_metrics_sha256=sha(output/'metrics.json'), top1_csv_sha256=sha(HERE/'top1.csv'),
        files={p.name: sha(p) for p in [HERE/'raw-metrics.csv', HERE/'relative-change-percent.csv',
            HERE/'absolute-difference.csv', HERE/'relative-change.png', HERE/'relative-change.pdf']},
        finished_utc=utc(), **limitations()))
    write_json(HERE / 'run-status.json', dict(status='delivery_artifacts_generated_review_pending',
        execution_complete=False, scanned_candidates=1281167, **limitations()))
    print('DELIVERY ARTIFACTS GENERATED; VISUAL REVIEW AND PUBLICATION PENDING', flush=True)


if __name__ == '__main__':
    main()
