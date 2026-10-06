"""Audit existing epoch logs and validation summaries; no models or GPUs are used.

Figure contract: Do later improvements in the training objective translate to
better held-out reconstruction scores? Panel a establishes the complete training
trajectory; panel b tests its relationship to the fixed development-set score.
Quantitative vertical grid, Python/Matplotlib, 183 x 143 mm, minimum font 7 pt,
editable PDF/SVG text, all 500 epochs plus every available same-pool checkpoint.
One subject and one training seed; no invented CI or significance tests.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KEYS = ['PixCorr', 'SSIM', 'Alex(2)', 'Alex(5)', 'Incep', 'CLIP', 'Eff', 'SwAV']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def mean_window(rows, start, end):
    vals = [r['mean_loss'] for r in rows if start <= r['epoch'] <= end]
    assert len(vals) == end - start + 1
    return sum(vals) / len(vals)


def make_figure(out, audit_scripts):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    with (out/'training_epochs.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (out/'validation_metrics.csv').open() as f:
        val = list(csv.DictReader(f))
    x = np.array([int(r['epoch']) for r in rows])
    y = np.array([float(r['mean_loss']) for r in rows])
    # The smooth line is a trailing, full 20-epoch window. No padding or
    # extrapolation; first plotted point is epoch 20.
    smooth = np.convolve(y, np.ones(20)/20, mode='valid')
    vx = np.array([int(r['epoch']) for r in val])
    vy = np.array([float(r['S']) for r in val])
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'], 'font.size':8,
        'axes.labelsize':8, 'axes.titlesize':9, 'xtick.labelsize':7,
        'ytick.labelsize':7, 'legend.fontsize':7, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False,
        'axes.spines.right':False, 'savefig.dpi':300})
    fig, axes = plt.subplots(2, 1, figsize=(7.20472440945, 5.62992125984), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.145, top=.91, hspace=.46)
    for i, ax in enumerate(axes):
        ax.annotate(chr(97+i), xy=(0,1), xycoords='axes fraction',
                    xytext=(-27,12), textcoords='offset points',
                    fontsize=10, fontweight='bold', va='bottom')
        ax.axvline(100.5, color='#777777', ls='--', lw=.8, zorder=0)
        ax.axvline(400, color='#C47A15', ls=':', lw=1.0, zorder=0)
        ax.set_xlim(0, 515)
        ax.set_xticks([0,100,200,300,400,500])
        ax.grid(axis='y', color='#EAEAEA', lw=.6)
    axes[0].plot(x,y,color='#BBC2CA',lw=.65,label='Raw epoch mean',zorder=1)
    axes[0].plot(x[19:],smooth,color='#007F86',lw=1.8,label='Trailing 20-epoch mean',zorder=2)
    axes[0].set_ylim(.105,.117)
    axes[0].set_yticks([.105,.108,.111,.114,.117])
    axes[0].set_ylabel('Min-SNR weighted noise MSE')
    axes[0].set_title('Training loss: slow improvement, with stochastic variation',loc='left',pad=18)
    axes[0].legend(frameon=False,ncol=2,loc='upper right')
    axes[0].tick_params(labelbottom=True)
    axes[1].plot(vx,vy,'o-',color='#355F9E',lw=1.6,ms=3.8)
    axes[1].scatter([400],[vy[list(vx).index(400)]],s=38,color='#C47A15',zorder=4)
    axes[1].set_title('Fixed Val500: the selection score peaks at L400',loc='left',pad=18)
    axes[1].set_ylabel('S = (AN5 + Incep + CLIP) / 3 (%)')
    axes[1].set_ylim(63,89.5)
    axes[1].set_xlabel('Cumulative training epoch')
    axes[1].annotate('L400: 86.30', (400,vy[list(vx).index(400)]),
                     xytext=(-16,12),textcoords='offset points',fontsize=7,ha='right')
    axes[1].annotate('L500: 85.48', (500,vy[list(vx).index(500)]),
                     xytext=(-4,-16),textcoords='offset points',fontsize=7,ha='right')
    fig.text(.12,.055,'Dashed: LR changes from 1e-4 to 5e-5 after L100. Dotted: selected L400.',fontsize=7)
    fig.text(.12,.023,'Subject 1; seed 42; Val500 with 2 fixed candidates averaged. Lines connect observed scores.',fontsize=7)
    fig.align_ylabels(axes)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=out/'training_vs_validation.alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'training_vs_validation.{ext}', dpi=300)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--audit-scripts',type=Path,required=True)
    ap.add_argument('--plot-only',action='store_true')
    a=ap.parse_args(); root=a.repo.resolve(); out=Path(__file__).resolve().parent
    if a.plot_only:
        make_figure(out,a.audit_scripts); return
    inputs={}
    def record(rel):
        p=root/rel; inputs[rel]=sha(p); return p
    def read(rel):
        return json.loads(record(rel).read_text())
    trajectories=[('runs/subject01-author-fp32-seed42/epochs.jsonl',1,100,1e-4),
                  ('runs/subject01-author-fp32-seed42-lr5e-5-from100/epochs.jsonl',101,500,5e-5)]
    rows=[]
    for rel,lo,hi,lr in trajectories:
        raw=[json.loads(s) for s in record(rel).read_text().splitlines() if s.strip()]
        chosen=[r for r in raw if lo <= r['epoch'] <= hi]
        assert [r['epoch'] for r in chosen] == list(range(lo,hi+1))
        rows.extend([{**r,'learning_rate':lr,'source':rel} for r in chosen])
    assert [r['epoch'] for r in rows]==list(range(1,501))
    assert all(math.isfinite(r['mean_loss']) for r in rows)
    assert all(r['global_optimizer_updates']==r['epoch']*532 for r in rows)
    assert all(r['unique_samples']==8500 and r['padding_samples']==4 for r in rows)
    assert len({r['image_order_sha256'] for r in rows})==500
    write_csv(out/'training_epochs.csv',rows)
    val_sources={e:f'reports/epoch-{e}/val500/metrics.json' for e in [20,50,70,80,100]}
    val_sources.update({110:'reports/lr-half-from100/L/val500/metrics.json',
        120:'reports/lr-half-to120/val500/metrics.json',
        130:'reports/l130-val500-followup/val500/metrics.json',
        140:'reports/lr-half-to140/epoch-140/val500/metrics.json'})
    val_sources.update({e:f'reports/lr-half-to{300 if e<=300 else 500}/epoch-{e}/val500/metrics.json'
                        for e in [200,250,300,400,450,500]})
    ref_ids=[int(i) for i in record('configs/val_ids.txt').read_text().split()]
    val=[]
    for e,rel in sorted(val_sources.items()):
        d=read(rel)
        assert d['n']==500 and d['image_ids']==ref_ids and d['candidate_selection'] is False
        assert len(d['candidate_metrics'])==2
        assert all(abs(d['mean'][k]-sum(m[k] for m in d['candidate_metrics'])/2)<1e-10 for k in KEYS)
        assert abs(d['S']-sum(d['mean'][k] for k in ['Alex(5)','Incep','CLIP'])/3)<1e-10
        val.append({'epoch':e,'optimizer_updates':e*532,
                    'loss_last20_epochs':mean_window(rows,e-19,e),**d['mean'],'S':d['S'],'source':rel})
    write_csv(out/'validation_metrics.csv',val)
    intervals=[(1,20),(81,100),(101,110),(491,500),(301,350),(351,400),(401,450),(451,500)]
    windows=[{'start_epoch':s,'end_epoch':e,'mean_loss':mean_window(rows,s,e)} for s,e in intervals]
    write_csv(out/'loss_windows.csv',windows)
    final=read('reports/lr-half-to500/training-final-audit.json')
    assert final['completed_epochs']==500 and final['optimizer_updates']==266000 and final['skipped_updates']==0 and final['exit_code']==0
    selection=read('reports/lr-half-to500/selection.json')
    control=read('reports/brain-correspondence-l120/comparison.json')
    controls=[]
    for c in ['correct','permuted']:
        d=control['summaries'][c]['val500']; assert d['image_ids']==ref_ids
        controls.append({'condition':c,'epoch':120,**d['mean'],'S':d['S']})
    write_csv(out/'existing_l120_brain_control.csv',controls)
    for rel in ['train_brain_adapter.py','brain_adapter/loss.py','brain_adapter/model.py',
                'brain_adapter/dataset.py','decode_brain_adapter.py','configs/stage1.json',
                'scripts/evaluate_stage.py','scripts/lr_half_evaluate.py','scripts/lr_half_to500_evaluate.py',
                'reports/brain-correspondence-l120/README.md']:
        record(rel)
    same_test='benchmarks/subj01-v1/imagenet-retrieval/local-fixed-v1/scoring-correction-20261006/raw-metrics.csv'
    with record(same_test).open(encoding='utf-8') as f:
        test=list(csv.DictReader(f))
    selected_test=[r for r in test if r['method'] in ['NeuroAdapter L400','MindEye1 K1','Brain-Diffuser K1','MindBridge(s) K1']]
    l300=read('reports/lr-half-to300/test982/metrics/metrics.json')
    l400=read('reports/lr-half-to500/test982/metrics/metrics.json')
    for d in [l300,l400]:
        assert d['n_eval']==982 and d['K']==1 and d['selection']=='none'
        assert d['eval_ids_hash']=='4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441'
    assert l300['metric_sha256']==l400['metric_sha256']
    rename={'Alex(2)':'AN2','Alex(5)':'AN5'}
    l400_table=next(r for r in selected_test if r['method']=='NeuroAdapter L400')
    assert all(abs(float(l400_table[rename.get(k,k)])-l400['mean'][k])<1e-12 for k in KEYS)
    selected_test.insert(0,{'method':'NeuroAdapter L300',
        **{rename.get(k,k):v for k,v in l300['mean'].items()},
        'disclosure':'Previously viewed standard982; not independent blind test',
        'n_eval':982,'encoder_development_overlap':'not applicable to this decoder comparison',
        'result_scope':'Existing K1 outputs; same frozen metric and ID hash'})
    write_csv(out/'existing_test982_context.csv',selected_test)
    by_epoch={v['epoch']:v for v in val}
    facts={
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      'audit_utc':datetime.now(timezone.utc).isoformat(),
      'new_training_updates':0,'new_decoded_images':0,'new_model_evaluations':0,
      'epoch_count':len(rows),'training_updates':266000,'skipped_updates':0,
      'validation_epochs':list(by_epoch),'all_validation_ids_and_candidate_means_verified':True,
      'val_ids_sha256':inputs['configs/val_ids.txt'],
      'loss_first10_low_lr':mean_window(rows,101,110),'loss_last10':mean_window(rows,491,500),
      'loss_low_lr_first10_to_last10_relative_pct':100*(mean_window(rows,491,500)/mean_window(rows,101,110)-1),
      'loss_351_400':mean_window(rows,351,400),'loss_451_500':mean_window(rows,451,500),
      'loss_late50_relative_pct':100*(mean_window(rows,451,500)/mean_window(rows,351,400)-1),
      'validation_S_400_to500_pp':by_epoch[500]['S']-by_epoch[400]['S'],
      'l120_correct_minus_permuted_S_pp':controls[0]['S']-controls[1]['S'],
      'missing':['stepwise loss history','gradient norms/clipping rates','fixed-noise validation loss',
                 'loss by timestep','token norms and parameter-update norms'],
      'input_sha256':inputs,
      'scope':'Existing records only; repeated internal validation use; no independent blind-test claim.',
      'figure':{'backend':'Python/Matplotlib','width_mm':183,'height_mm':143,
          'panels':['a: all epoch losses plus trailing full 20-epoch mean',
                    'b: all 15 existing same-pool Val500 S scores'],
          'uncertainty':'No error bars: one subject, one training seed, fixed candidate average; no significance test',
          'loss_axis':'Zoomed 0.105 to 0.117; absolute values labelled; not a zero-origin plot'}}
    save(out/'audit.json',facts)
    make_figure(out,a.audit_scripts)
    def metric_table(vs):
        text='|轮次|末20轮loss均值|PixCorr ↑|SSIM ↑|AN2 ↑|AN5 ↑|Incep ↑|CLIP ↑|Eff ↓|SwAV ↓|S ↑|\n'
        text+='|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for v in vs:
            text+=f"|{v['epoch']}|{v['loss_last20_epochs']:.6f}|"+'|'.join(f'{v[k]:.5f}' for k in KEYS)+f"|{v['S']:.5f}|\n"
        return text
    report=f'''# NeuroAdapter 连续1–500轮：训练损失与效果诊断

本次只分析已有日志、已有验证评分和已有对照，新增训练更新、生成图片、模型评价均为0。模型是Subject1、seed42的作者路线；epoch1–100使用LR1e-4，epoch101–500使用5e-5，最终500轮/266000更新/0 skipped，原选中权重L400和best均未修改。

## 现在已有的结论

1. **有完整500轮训练loss，但没有完整逐step/梯度历史。** 使用逐epoch `mean_loss`，不是控制台最后一个microbatch的 `avg_train_loss`。原高LR文件取1–100；其101–110属于高LR对照，明确排除；再接低LR101–500。500轮无缺失、loss全部有限、更新计数均为epoch×532、每轮8500唯一样本加4条DDP补齐、500个采样顺序SHA不同。已记录epoch loss未见发散，累计0 skipped。最初续训有过loader deepcopy启动错误并修复；这不等于整条训练从未发生工程故障。

2. **后期继续压低训练loss，没有带来持续的验证改善。** L351–400均值{facts['loss_351_400']:.8f}，L451–500为{facts['loss_451_500']:.8f}，仅下降{-facts['loss_late50_relative_pct']:.3f}%。而相同val500的S由L400的{by_epoch[400]['S']:.5f}降到L450的{by_epoch[450]['S']:.5f}、L500的{by_epoch[500]['S']:.5f}，L400→500下降{-facts['validation_S_400_to500_pp']:.5f}个百分点。支持暂停单纯堆轮数；尚不能据此证明过拟合或统计显著退化。

3. **这个loss不是图像忠实度。** 代码目标是 `mean(min(SNR,5)/SNR × noise_MSE)`（实现含数值epsilon），训练latent采样、扩散噪声、时间步与token dropout均随机。没有直接PixCorr、SSIM、CLIP或图像像素重建项。loss约0.11不能当作“只错11%”。早期val S从L20的{by_epoch[20]['S']:.2f}提高到L100的{by_epoch[100]['S']:.2f}，训练平均loss却变化很小，已经说明总loss是重建质量的粗指标。仅凭下降1–2%不能断言没学到。

4. **有学习脑信号的证据，问题不是已证实的“完全忽略fMRI”。** 既有L120同500图、同种子和同解码参数对照：正确脑输入S={controls[0]['S']:.5f}，打乱为{controls[1]['S']:.5f}，差{facts['l120_correct_minus_permuted_S_pp']:.5f}个百分点；八项方向均支持正确对应。500个供体一一置换、无自配，真实输入SHA已核对。这个证据只覆盖该L120与内部开发池；不能假装L400/500也跑过相同对照，也不能把“有输入依赖”等同于布局与细节已经正确。

## 曲线和原始数据

![500轮训练loss与固定验证S](training_vs_validation.png)

图a浅灰为全部500个真实epoch均值；青色为**向后20轮移动平均**，第一点在epoch20，不补齐、不外推。loss轴为0.105–0.117局部范围，实际差值应读刻度。图b为15个已经完成且ID顺序完全相同的val500检查点；连接线只帮助看趋势，未评估轮次没有新增分数。虚线表示100轮后降低LR，点线表示原选中L400。

val500每图2个固定候选、50步、CFG4、seed42、candidate batch1；两个候选各自在全500负例池评分后平均，未挑图。AN2/AN5/Incep/CLIP与S为百分数标度的两两识别指标，不是500类top1准确率。模型只有一条seed42训练轨迹，图中不制造置信区间或显著性；验证集被反复用于开发。

- [完整500轮loss与更新记录](training_epochs.csv)：包含epoch、loss、耗时、LR、样本数、顺序SHA与来源。
- [全部15个验证点、完整八指标及S](validation_metrics.csv)。
- [固定区间loss均值](loss_windows.csv)、[既有L120输入打乱对照](existing_l120_brain_control.csv)。
- [可编辑PDF](training_vs_validation.pdf)、[SVG](training_vs_validation.svg)、[作图和核验代码](build_diagnostics.py)、[输入SHA与核验记录](audit.json)。

## 统一口径的完整验证表

所有行均同val500、同K2协议。loss列统一使用截至该轮的20轮均值，不是验证loss。

{metric_table(val)}

L400→500各指标并非全部变差：AN5从86.48277到86.70040，SSIM从0.27592到0.28320；但CLIP从87.39800降至84.96353，PixCorr也下降。综合S下降有具体组成，不能只报一个分数，也不能声称“所有指标恶化”。

## 差效果具体体现在哪里

标准982 K1的L400为PixCorr 0.09638、SSIM 0.27739、AN2 78.06605、AN5 88.05751、Incep 85.07851、CLIP 86.39870、Eff 0.77114、SwAV 0.45502。已有MindEye1同982 K1的PixCorr为0.36503、AN2 96.55533、CLIP 94.30763；L400在像素对应及这些特征指标都有差距。Brain-Diffuser的PixCorr/SSIM更高，但Incep/CLIP更低，提示不同方法有不同权衡。这里描述性能模式，不把模型间差异当成某模块的因果归因。

L300→L400同982结果中PixCorr仅0.09311→0.09638、SSIM反而0.29336→0.27739；S选优不包含PixCorr和SSIM，因此它没有保证结构/像素指标同时提升。**现有证据更符合“学到了部分语义条件，但位置、姿态、细节恢复仍弱；后期续训未带来持续提升，出现平台迹象”这一描述。** 这仅覆盖现有固定路线和观察区间，不是证明网络能力有绝对上限。

以上同982对照读取已经校正并核验的主表，并补入已核对ID hash、K1和冻结指标SHA的L300原结果，汇总在[既有982对照数据](existing_test982_context.csv)。没有混入原100k的50图best8或MindEye best16。epoch100不等于100k更新。先前ImageNet检索浮点越界触发/255是另一个已经修正的评分问题；L400已有实际输入范围核验通过，不能用那次baseline错误解释L400性能。

## 哪些原因已检查，哪些仍不能定论

|问题|当前证据|判断边界|
|训练中断、NaN、跳更新导致没训练到位|500轮/266000更新/0skipped、完整恢复审计和有限loss|不支持这一解释；不等于优化已最优|
|训练与推理误用了不同parcel/输入处理|同dataset按ID平均重复，推理strict加载与parcel顶点断言|此次代码审阅未发现可认定的错配，不等于所有科学选择已最优|
|模型完全不看脑输入|L120同池输入置乱使S下降33.49646pp|可排除该L120对照中的完全无效；不能直接外推L400|
|过拟合|后期train loss略降、验证S下降|有迹象但未证实：缺同定义、固定随机量的train/val loss|
|训练目标与期望指标不完全一致|训练仅加权噪声MSE，选优S仅3项语义特征|目标定义是事实；对差效果贡献多大尚未量化|
|优化/表达/条件强度瓶颈|后期收益小，LR101–500固定，随机TokenDropout|都是待检验假设，不能宣布某一个是原因|

模型此路径为200个parcel独立线性投影生成200个条件token；配置名 `num_decoder_queries=50` 不代表此路径实际50 token。TokenDropout对每样本抽U(0,1)保留率，平均保留约100/200，但不是每张固定保留一半。推理使用全部token和全零条件做CFG4。这些是现有训练设计，尚无证据把它们直接叫作bug。

## 下一步最有判别力的诊断（本次未执行）

先补**固定噪声的train/val去噪loss**，而不是接着训600轮：对现有L120/L300/L400/L500，固定训练子集、val500及每个ID的latent、noise、timestep，在eval模式下同时计算加权/未加权MSE并按噪声时间步分组。四个权重只做前向，新增训练更新可保持0。

- 若训练集持续改善而验证集恶化，才获得更直接的过拟合证据。
- 若训练集和验证集都停滞，应进一步检查优化幅度和表达瓶颈。
- 若总loss变化小但高噪声阶段变差，可定位“总均值掩盖哪个难度段”的问题。
- 如需确认L400后脑条件是否衰减，在同一固定前向诊断里比较真脑/打乱脑/零条件，不重复已完成的L120全500图生成实验。token尺度、条件与无条件预测差可进一步定位环节。

历史未记录逐step loss、梯度范数/裁剪比例、固定噪声验证loss、按timestep loss及参数更新量；不能事后编造曲线。未来若获准续训，再将这些量写入日志。当前先保持L400与所有历史产物不变。

## 源码与可信度

- `train_brain_adapter.py:343–388`：随机训练样本/时间步和MinSNR目标；`:435–505`：双rank loss、梯度有限性检查和epoch均值。
- `brain_adapter/loss.py:18–23`：MinSNR权重；`brain_adapter/model.py:138–149`：parcel投影。
- `scripts/evaluate_stage.py`、`scripts/lr_half_evaluate.py`、`scripts/lr_half_to500_evaluate.py`：现有统一验证协议。
- `reports/lr-half-to500/training-final-audit.json`：已完成训练的事实审计；`selection.json`：原S选优。
- `reports/brain-correspondence-l120/README.md`：已做的正确/打乱输入对照及其局限。

原始输入SHA保存在audit.json，本次没有训练、下载模型、改权重、改参数、改旧评分或读取权重到GPU。报告将已证实的观察与尚未做的归因实验分开；不声称独立盲测，不用论文数字填充本地结果。
'''
    (out/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    assert all(sha(root/p)==h for p,h in inputs.items()), 'An input changed during audit'
    print(json.dumps({k:v for k,v in facts.items() if k not in ['input_sha256','figure']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
