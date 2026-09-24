from pathlib import Path
import json, hashlib, subprocess, statistics, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd()
out=root/'reports/evidence-freeze-seed43-plan'
out.mkdir(exist_ok=True)
source=root/'reports/hl-content-replication-l120'
c=json.loads((source/'comparison.json').read_text())
paths=['configs/stage1.json','configs/lr-half-to120.json','configs/train_ids.txt','configs/val_ids.txt','reports/best_so_far.json','reports/ibbi-roi-l120-pilot/roi_mapping.json','reports/hl-content-replication-l120/protocol.json','reports/hl-content-replication-l120/target_ids.txt','reports/hl-content-replication-l120/atlas32_ids.txt','reports/hl-content-replication-l120/comparison.json','reports/hl-content-replication-l120/EVIDENCE_SUMMARY.md','runs/subject01-author-fp32-seed42/epochs.jsonl','runs/subject01-author-fp32-seed42-lr5e-5-from100/epochs.jsonl']+[f'reports/hl-content-replication-l120/donor-{i}/map.json' for i in (1,2,3)]
locks={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
rows=[json.loads(x) for x in (root/paths[11]).read_text().splitlines() if x.strip()]
median=statistics.median(x['seconds'] for x in rows)
proposal={'status':'DESIGN_ONLY_NOT_STARTED','training_seed':43,'subject':1,'endpoint_epochs':120,'expected_updates':63840,'learning_rate_by_epoch':{'1-100':0.0001,'101-120':0.00005},'new_training_runs':1,'formal_generated_candidates':2000,'entry_check_candidate_cap':8,'target_donor_and_atlas_files_sha256':locks,'training_seconds_per_epoch_historical_median':median,'estimated_training_hours_excluding_io':median*120/3600,'existing_formal_runs':['subject01-author-fp32-seed42','subject01-author-fp32-seed42-lr5e-5-from100'],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'no_new_training_or_inference_started':True}
(out/'protocol-proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')
figure_rows=[]
fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
for ax,effect in zip(axes,['D','R']):
    for i,label in enumerate(['Aggregate','Donor 1','Donor 2','Donor 3']):
        values=c['aggregate'][effect] if i==0 else c['by_batch'][str(i)][effect]
        r=next(v for v in values if v['metric']=='S')
        figure_rows.append({'effect':effect,'group':label,**r})
        ax.errorbar(r['delta'],3-i,xerr=[[r['delta']-r['low']],[r['high']-r['delta']]],fmt='o',color='#146b87' if i==0 else '#777777',capsize=4)
    ax.set_yticks(range(4),['Donor 3','Donor 2','Donor 1','Aggregate'])
    ax.axvline(0,color='black',lw=.6)
    ax.set_title('D = AA - BA' if effect=='D' else 'R = AB - BB')
    ax.set_xlabel('S difference (percentage points)')
    ax.grid(axis='x',alpha=.2)
fig.suptitle('Fixed L120: content intervention replication')
fig.text(.5,.01,'Aggregate: nominal 97.5% CI; batches: 95% CI. Reused development set; no cross-seed inference.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.93))
fig.savefig(out/'primary-effects.png',dpi=180)
fig.savefig(out/'primary-effects.pdf')
with (out/'primary-effects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(figure_rows[0]));w.writeheader();w.writerows(figure_rows)
(out/'README.md').write_text('''# 固定L120证据收束与跨seed复核方案

状态：本轮仅整理既有证据和冻结待启动方案；没有新训练、生成、重新评分或自动任务。服务器当前正式权重只有seed42主线及其L支线；smoke目录不是第二个独立训练seed。读取时GPU计算进程为空。此前暂停与PROVEN_INFEASIBLE结论保留。

![主要结果](primary-effects.png)

图直接读取既有comparison.json，未重算统计。聚合97.5%与单批95%名义区间不能混为相同置信水平。底层数据见[CSV](primary-effects.csv)，可导出[PDF](primary-effects.pdf)。

## 当前能说什么

固定L120下，HL位置的内容换走与补回对目标相关语义指标及CLIP内容方向有一致的平均作用，在新100目标及三批供体中重复出现。聚合D=20.423120（97.5%区间17.290636～23.490509），R=20.546577（17.533649～23.537640）。这不是重建质量改善实验，也不是人脑因果或功能专一性证据。

D的SSIM点估计为负且区间跨零；25/34个目标的D/R在供体间发生符号变化。AA本身有整类错误，AB仍低于AA，补回不完全、混合内容和无法判断的案例均保留。所有八项、候选趋势和负面结果见[原报告](../hl-content-replication-l120/README.md)，[24页AI审阅](../hl-content-replication-l120/AI_REVIEW.md)和[全部目标图册](../hl-content-replication-l120/ALL_TARGETS.md)。本轮未新增图像标签或重新宣称审阅。

完整五环证据链见[既有证据汇总](../hl-content-replication-l120/EVIDENCE_SUMMARY.md)：整脑对应性→HL遮蔽先导→严格匹配不可行→单供体内容干预→新目标三供体复核。各自评价池和区间单独解释，不合并成总效应。右半球空间匹配PROVEN_INFEASIBLE没有被内容干预解决。

## 下一步决策

优先检验当前现象能否在一个预先指定的新训练seed复现。完整方案见[PLAN](PLAN.md)，配置及旧文件哈希见[protocol-proposal.json](protocol-proposal.json)。seed43属于新初始化/新训练随机流；从seed42 checkpoint改seed只会产生共享训练历史的分支，不能满足该问题。

此方案尚非可执行入口，未修改训练器或删除旧run_id/LR保护。待新增计算预算确认后，复用作者循环增加独立显式入口并验证；不提供一个现在会误运行旧主线的命令。
''',encoding='utf-8')
(out/'PLAN.md').write_text(f'''# Seed43有限复核计划（设计已固定，未启动）

## 问题与唯一新训练seed

问题：同一subject01、数据划分、parcel映射和既定训练配方下，HL内容D/R能否在第二个训练随机种子重复。只增加seed43，不搜索seed，不用结果选择epoch。原seed42权重、best、历史报告全部只读。

新run建议subject01-author-fp32-seed43-lrhalf-at100；从与seed42相同的预训练基础模型和作者随机初始化机制开始。训练seed43影响初始化、训练顺序和训练噪声；解码仍保留历史目标ID派生随机流，不能顺带改解码seed。

## 训练预算与配方

总计120真实epoch，按历史每epoch532次更新预计63840更新；前100轮LR1e-4，完整保存后只改LR为5e-5继续20轮。第100轮53200、第110轮58520、第120轮63840；实际进度与跳步均审计。第100轮保留AdamW/RNG/loader连续状态，改变LR前后验证其余状态未变，不重新seed。这里是预先固定的一次LR切换，不添加scheduler，不根据分数调LR。

双4090、FP32/mixed_precision=no、现有TF32/backend、micro4 accum2、原8500/500、作者模型、token dropout、MinSNR5、AdamW其他设置及预处理全部继承锁定配置。保留100/110/120完整状态和导出，并使用既有滚动恢复保护；不覆盖seed42，不调用主线best汇总。仅最终120权重进入内容复核，中途不按开发集分数选模型。

历史主线epoch日志中位数为{median:.2f}秒，120轮训练粗估{median*120/3600:.1f}小时（双GPU同时占用，约{median*120/3600*2:.1f} GPU小时）；不含初始化、保存、评价、图册与异常重试，不能当作完成时刻保证。此预算远大于在现有L120上再做一次推理。

## 固定推理预算

直接复用上一轮锁定的新100目标、三批供体映射、78个HL索引和前32图册ID，不重新划分。旧seed42输出只作历史对照，不能作为seed43的AA。

新seed43 AA生成100×2=200；三批BA/AB/BB共100×3×3×2=1800；正式共2000候选。接入恒等及全供体复现检查最多8候选，总上限2008。不增加全500评价、额外train16、第四供体或第二个训练seed。50步CFG4、batch1和原解码随机过程不变；E后ImageProj前仅同parcel内容替换。

## 评价和判定

每候选在同一新100目标GT池调用原八指标，再平均候选；CLIP方向量沿用现有实现。D=AA−BA、R=AB−BB，S=(Alex5+Incep+CLIP)/3。每目标先平均2候选、3供体，再以100目标块做10000次配对bootstrap，固定seed2026092409，主S各97.5%名义区间，辅助指标及单批95%。同时交付三批及两候选趋势，Eff/SwAV低好。

只有新seed的D/R主S下界均>0且三批点估计均正，才描述“在第二个预定seed方向复现”；区间跨零说未确认，不等于无效。不同seed的结果各自报告，不把两个seed的候选拼成独立样本，也不以图像bootstrap声称seed总体显著。仅两个seed不能给出稳定的跨seed总体推断；同一开发池更不能称独立test或跨subject泛化。

固定120而非开发集最优epoch，与历史自适应选择的L120存在选择不对称，须明确披露。无论结果如何，不事后挑另一个epoch/seed挽救。混合E可能离分布、HL空间特异性未被证明、人脑因果不可推断等限制全部保留。

## 图册与停止

新32目标的三批×两候选×四页，共24页；包含目标GT、供体GT、四条件全部候选。AI非盲审阅，改善/失败/混合/未迁移一并呈现，链接旧图册，不新增大规模标签表。

数值非有限、恢复或来源/顺序审计失败即保留最后COMPLETE并报告；不得换seed、降LR、改变预算补救。正常训练不因开发分数波动筛掉新seed。120训练和规定推理/报告完成后退出，检查GPU释放，不继续150/300或下一实验。

## 启动前尚待完成

本轮只完成设计：需要明确接受上述新训练和推理预算，随后再实现最小参数化入口、核实存储余量与GPU、锁定代码/初始化/配置及恢复审计。旧脚本带有seed42/run_id/目标保护，不能直接拼一个旧命令冒充seed43启动。当前没有训练启动命令、PID或自动跟进任务。
''',encoding='utf-8')
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in locks.items())
print(json.dumps({'status':'design_only','historical_files_unchanged':len(locks),'training_hours_estimate':median*120/3600,'outputs':sorted(p.name for p in out.iterdir())}))