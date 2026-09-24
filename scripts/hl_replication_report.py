"""Render locked replication statistics without changing evaluation or historical reports."""
import json
from pathlib import Path
import pandas as pd
import hl_content_replication as study
R=study.REPORT
def read(p):return json.loads(p.read_text())
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
def fmt(x):return f'{x:.6f}'
def ci(r):return f"{fmt(r['delta'])} [{fmt(r['low'])}, {fmt(r['high'])}]"
def main():
    c=read(R/'comparison.json');done=read(R/'completion.json')
    assert done['compute_complete'] and done['formal_candidates']==1800
    lines=['# L120固定HL内容干预：新目标与三供体复核','',
    '本轮固定L120、原78个HL位置及作者decoder；新增训练更新为0。原100目标单供体报告独立保留。本轮不是独立test，不解决严格空间匹配不可行问题，也不证明人脑因果或功能专一性。','',
    '## 主要结果','']
    primary={k:next(r for r in c['aggregate'][k] if r['metric']=='S') for k in ['D','R']}
    signs=all(next(r for r in c['by_batch'][str(m)][k] if r['metric']=='S')['delta']>0 for m in range(1,4) for k in ['D','R'])
    if all(r['low']>0 for r in primary.values()) and signs:
        lines+=['在新增内部开发目标和三个预先固定供体批次上，换走损失与补回收益的方向得到重复支持。该结论限于固定模型、配对和评价池；不表示重建已修好。','']
    else:
        lines+=['本轮未同时满足两个主要区间下界大于0且三个批次点估计均为正的预定描述条件。下面完整保留各方向及批次差异；未确认不等于无效。','']
    lines+=table(['主要S终点','点估计与名义97.5%区间（百分点）'],[[k,ci(v)] for k,v in primary.items()])
    lines+=['D=AA−BA，R=AB−BB。先平均每图两个候选，再等权平均三个供体，最后对100个目标块汇总；10000次配对bootstrap，seed2026092409，共用重采样索引。','',
    '## 同一新100目标池的全部八项指标','',
    '每个候选分别调用作者指标后平均。Eff/SwAV越低越好，其余越高越好。AA仅评分一次；三批引用同一AA，不增加样本量。','']
    keys=study.KEYS+['S'];rows=[]
    for m in range(1,4):
        for label in ['AA','BA','AB','BB']:
            x=c['by_batch'][str(m)]['summaries'][label]
            rows.append([str(m),label]+[fmt(x['mean'][k] if k!='S' else x['S']) for k in keys])
    lines+=table(['批次','条件']+keys,rows)
    lines+=['## 每批D/R、候选与供体异质性','',
    '以下每批区间全部为辅助名义95%，不新增主检验。三批来自一次无放回划分，不是三个独立受试者或三份独立总体抽样。','']
    for m in range(1,4):
        lines += [f'### 供体批次{m}','']
        lines += table(['指标','D（95%）','R（95%）'],[[a['metric'],ci(a),ci(b)] for a,b in zip(c['by_batch'][str(m)]['D'],c['by_batch'][str(m)]['R'])])
        lines += table(['条件','cos_target','cos_donor','q'],[[k]+[fmt(v[z]) for z in ['cos_target','cos_donor','q']] for k,v in c['by_batch'][str(m)]['direction_means'].items()])
        lines += [f'[逐图配对结果](donor-{m}/paired-results.csv) · [方向量逐候选CSV](donor-{m}/clip-direction-candidates.csv)','']
    cand=pd.read_csv(R/'candidate-contrast-trends.csv')
    lines+=['### 两个候选各自的S方向','']
    lines+=table(['批次','候选','终点','S差值'],[[int(r.batch),int(r.candidate),r.contrast,fmt(r.S)] for r in cand.itertuples()])
    lines+=['[全部指标的逐候选趋势](candidate-contrast-trends.csv) · [100个目标的三供体范围及全部值](target-block-results.csv)','',
    '供体敏感性必须结合逐图范围而不是只看平均。AA的cos_donor与q已为每批供体分别重算；q不是作者CLIP识别率，不参与生成或选图。','',
    '## 聚合效应：全部八项及辅助方向量','']
    lines+=table(['指标','D','R','区间'],[[a['metric'],ci(a),ci(b),'97.5%（主要）' if a['metric']=='S' else '95%（辅助）'] for a,b in zip(c['aggregate']['D'],c['aggregate']['R'])])
    lines+=['原始Eff/SwAV差值保留符号：在D/R定义下，负值对应预期的距离改善方向。其他指标正值对应预期方向。','',
    '### 辅助D−R','']
    lines+=table(['指标','差值及95%区间'],[[x['metric'],ci(x)] for x in c['aggregate']['D_minus_R']])
    lines+=['D−R不解释为脑区协同；不使用恢复比例宣称脑信息占比。','',
    '## 全部固定图册与AI审阅','',
    '每页8个预定目标，列顺序为目标GT、供体GT、AA、BA、AB、BB。每批两个候选全部展示。','']
    for m in range(1,4):
        for k in range(2):
            lines += [f"- 批次{m}、候选{k}："+' / '.join(f'[第{p}页](donor-{m}/atlas/candidate-{k}-page-{p}.jpg)' for p in range(1,5))]
    lines+=['','[新100目标全部候选浏览索引](ALL_TARGETS.md)','']
    if (R/'AI_REVIEW.md').exists():lines+=['[AI非盲实际审阅记录](AI_REVIEW.md)。请结合失败、混合内容及不确定案例阅读数值。','']
    else:lines+=['24页固定图册的实际AI审阅尚未完成，本文件此时不能视为最终交付。','']
    lines+=['## 锁定、预算与退出证据','',
    '[开跑前PLAN](PLAN.md) · [协议](protocol.json) · [300行固定配对](all-target-donor-pairs.csv) · [模型锁](model-lock.json) · [HL索引](hl-indices.json) · [8张接入逐像素检查](noninterference.json)','',
    '[1800张正式生成清单](generation-manifest.json) · [逐条件输入及随机流审计](content-audits.json) · [完成与保护哈希](completion.json)','',
    '启动命令：`runtime/env/bin/python scripts/hl_content_replication.py`。各阶段实际命令及退出码保存在`logs/hl-replication-events.jsonl`；最终发布另附运行核验。','',
    '## 解释边界','',
    '这100张是上一轮HL混合之外的新目标，但已用于500图模型开发与其他实验，不是未见过的独立测试集。旧100池与新100池不混分，不把绝对S变化称为模型进步。','',
    '固定负例池、单个选择过的模型、单一训练seed和历史自适应开发限制推断。名义区间只描述当前固定目标及三供体块，不提供人脑总体、所有供体或所有采样噪声的无条件覆盖；不校正历史模型选择。','',
    '混合E可能离开训练分布，条件冲突与输出退化不能一律叫语义转移。AB未完全恢复AA及AA原本错误均保留；不把正结果解释为人脑因果或HL功能专一性。','',
    '[既有研究证据汇总](EVIDENCE_SUMMARY.md)。固定预算结束后停止，不追加供体、子ROI、权重或训练。','']
    lines += ['', (R/'INTERPRETATION.md').read_text(), '', '[实际退出、预算、433项保护及24页审阅核验](delivery-verification.json)']
    (R/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
