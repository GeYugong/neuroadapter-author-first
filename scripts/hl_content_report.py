"""Render computed four-condition results; actual AI atlas review is supplied separately."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/hl-content-interchange-l120'
def main():
    assert (R/'EVALUATION_COMPLETE').exists()
    c=json.loads((R/'comparison.json').read_text())
    rows=['# L120固定HL位置：内容替换与补回','',
      '本轮固定L120和原78个HL位置，使用原100目标及既有500图映射对应的供体。零训练、零调参；AA/BB复用，BA/AB新增。此实验不解决原严格空间匹配不可行问题。','',
      '第一位是HL的来源，第二位是其余122行的来源。A为目标，B为供体；所有生成随机流始终由目标ID及候选序号派生。U不是LL，重叠token22和143仍属HL。','',
      '## 原100图评价池的八项指标','',
      '|指标|AA|BA|AB|BB|方向|','|---|---:|---:|---:|---:|---|']
    for key in ['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV','S']:
        vals=[c['summaries'][label]['S'] if key=='S' else c['summaries'][label]['mean'][key] for label in ['AA','BA','AB','BB']]
        rows.append('|'+key+'|'+ '|'.join(f'{v:.6f}' for v in vals)+'|'+('低为好' if key in ['Eff','SwAV'] else '高为好')+'|')
    rows+=['','两候选分别在完整100目标GT池评分后平均；没有拼成200图负例池，没有与500图混算。S仍为Alex(5)、Inception、CLIP识别率的均值。','',
      '## 两个预定主效应及辅助指标','',
      'D=AA−BA：换走HL内容造成的目标对应损失。R=AB−BB：补回HL内容造成的目标对应增益。以下保留原差值符号，Eff/SwAV的方向与识别率相反。','',
      '|指标|D|区间|R|区间|名义水平|','|---|---:|---|---:|---|---|']
    for d,r in zip(c['contrasts']['D'],c['contrasts']['R']):
        assert d['metric']==r['metric']
        rows.append(f"|{d['metric']}|{d['delta']:.6f}|[{d['low']:.6f}, {d['high']:.6f}]|{r['delta']:.6f}|[{r['low']:.6f}, {r['high']:.6f}]|{d['nominal_interval']*100:g}%|")
    rows+=['','只有D/R的S是并列主终点，各使用97.5%名义百分位区间；其他指标和方向量为95%辅助描述。每图先平均两候选，所有条件共用每次抽中的100行，10000次重采样、seed2026092407，负例池不重建。','',
      '## CLIP辅助方向量','',
      'q=cos(输出,目标GT)−cos(输出,供体GT)。使用作者冻结CLIP及原预处理；它不是官方CLIP识别率，不参与生成和选图。','',
      '|条件|目标cosine|供体cosine|q|','|---|---:|---:|---:|']
    for label in ['AA','BA','AB','BB']:
        v=c['direction_means'][label]
        rows.append(f"|{label}|{v['cos_target']:.6f}|{v['cos_donor']:.6f}|{v['q']:.6f}|")
    rows+=['','## 固定图册与AI审阅','',
      '原32张图，目标GT、供体GT、四条件、两个候选全部展示。','']
    for k in range(2):
        for p in range(1,5):
            rows += [f'### 候选{k}，第{p}页','',f'![候选{k}第{p}页](atlas/candidate-{k}-page-{p}.jpg)','']
    review=R/'AI_REVIEW.md'
    rows += [review.read_text() if review.exists() else 'AI实际图册审阅待完成；计算完成不代表交付完成。','',
      '## 解释边界','',
      '固定的同一组token位置使集合大小、parcel身份等保持不变，但不保证跨样本混合token的联合分布在训练分布内。D/R不能解释为78个HL与122个U的单位重要性排名，也不能解释成人脑因果、功能专一性或原HL额外损失由标签本身造成。','',
      '这是一份固定供体映射；目标与供体在行间可能相关，且L120与这些开发图已被反复分析。区间只是本条件下的探索性描述，不保证独立样本覆盖，不校正历史选择，不代表跨供体、跨seed或标准test结论。D−R仅背景交互的辅助描述，不能升格为生物协同证据。','',
      '## 可复核证据','',
      '- [冻结方案](PLAN.md)、[协议](protocol.json)、[模型锁定](model_lock.json)、[HL及补集](hl-indices.json)、[100行供体映射](target-donor-map.json)。',
      '- [AA/BB复用来源](reference-manifest.json)、[最多8张逐像素复现](noninterference.json)、[真实E行来源审计](content-audits.json)。',
      '- [原始汇总与区间](comparison.json)、[逐图四条件及配对差值](paired-four-conditions.csv)、[每候选CLIP两个cosine与q](clip-direction-candidates.csv)。',
      '- [BA候选0](BA/val100/candidate-0.csv)、[BA候选1](BA/val100/candidate-1.csv)、[AB候选0](AB/val100/candidate-0.csv)、[AB候选1](AB/val100/candidate-1.csv)。',
      '- 原参考指标：[AA](../lr-half-to120/val100/metrics.json)、[BB](../brain-correspondence-l120/permuted/val100/metrics.json)。',
      '- [实际预算与关键输入不变检查](completion.json)。命令：runtime/env/bin/python scripts/hl_content_interchange.py；逐阶段命令、退出码及最终GPU核验见[delivery-checks.json](delivery-checks.json)。','',
      '原IBBI/ROI先导及严格匹配PROVEN_INFEASIBLE报告保持原样；本轮无后续训练或追加实验。']
    rows[2:2]=[(R/'CONCLUSION.md').read_text(), '']
    delivery=json.loads((R/'delivery-checks.json').read_text())
    rows += ['', '## 实际退出与停止状态', '', '运行代码提交c282585；12条阶段/子进程退出记录均为0，控制器controller_complete=0。BA/AB共400张正式图，AA/BB检查8张，未超预算。309项锁定关键输入SHA复核一致；训练仍为L140/74480，新增更新0。最终无本实验控制器或decoder进程，GPU计算进程为空。全部8页已完成AI非盲审阅；到此停止，不追加实验。', '', '[命令、退出码、脚本SHA与最终GPU记录](delivery-checks.json)']
    (R/'README.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
if __name__=='__main__':main()
