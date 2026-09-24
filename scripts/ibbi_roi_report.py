"""Publish numeric tables from completed author scores; visual review stays explicitly separate."""
import json
import numpy as np
import pandas as pd
from ibbi_roi_prepare import ROOT,RUN,REPORT,WORK
from ibbi_roi_pilot import KEYS,CONDITIONS
import lr_half_evaluate as old
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(x) for x in row)+'|' for row in rows])
def main():
    assert (REPORT/'EVALUATION_COMPLETE').exists()
    c=old.read(REPORT/'comparison.json');mapping=old.read(REPORT/'roi_mapping.json')
    complete=old.read(REPORT/'completion.json');stats=old.read(REPORT/'mapping-characteristics.json')
    events=[json.loads(l) for l in (ROOT/'logs/ibbi-roi-l120-events.jsonl').read_text().splitlines()]
    old.save(REPORT/'execution-events.json',events)
    vals=c['summaries'];names=['BASE']+[n for n in CONDITIONS if n in vals]
    summary_rows=[[n]+[f"{vals[n]['mean'][k]:.6f}" for k in KEYS]+[f"{vals[n]['S']:.6f}"] for n in names]
    with np.load(REPORT/'attention/all-brain-attention.npz') as a:b=a['brain_attention']
    assert b.shape==(64,50,200)
    pd.DataFrame({'token_index':range(200),'mean_attention_share':b.mean(axis=(0,1))}).to_csv(REPORT/'attention/parcel-mean-share.csv',index=False)
    bygroup=pd.DataFrame({'denoising_step':range(1,51)})
    for g,ts in mapping['groups'].items():bygroup[g]=b[:,:,ts].sum(axis=-1).mean(axis=0)
    bygroup.to_csv(REPORT/'attention/group-share-by-step.csv',index=False)
    lines=['# L120固定权重：IBBI与ROI遮蔽先导实验','',
        '本实验新增训练更新为0。L120、原parcel映射、历史权重/输出及主线best均保持原样。结果仅描述这一固定模型的输入通路依赖；注意力不是人脑因果贡献，也不表示模型已达到论文重建水平。','',
        '## 固定范围与可复核产物','',
        '- [模型锁](model_lock.json)、[协议](protocol.json)、[执行计划](PLAN.md)、[实际命令与退出事件](execution-events.json)。',
        '- [两图四候选逐像素不干扰/空mask检查](noninterference.json)，四条检查轨迹计入32图双候选的64条总预算。',
        '- [完整parcel/功能标签重叠表](parcel_roi_mapping.csv)、[冻结ROI/随机集合](roi_mapping.json)、[SNR/顶点/空间邻接描述](mapping-characteristics.json)。',
        '- [全部64条IBBI注意力图与定量数组](attention/README.md)、[全部条件原32图两候选并排图册](all-condition-atlas/README.md)。',
        '- [逐图D/E与区间汇总](comparison.json)、[保护与零更新核验](completion.json)。','',
        '## ROI映射和对照限制','',
        'Subject1官方native功能标签通过NSD逐半球white-to-fsaverage最近邻索引变换转入163842顶点空间，按L120保存的200个parcel实际有效顶点计算；每个原始标签严格重叠>0.5，padding不进分母，未先合并标签。没有采用作者绘图默认0.75或dataset中的>=比较。','',
        f"实际LL={len(mapping['groups']['LL'])}、HL={len(mapping['groups']['HL'])}，LL/HL重复token={mapping['LL_HL_intersection']}。保留多标签，不凑论文50/53数量。","",
        table(['集合','parcel','LH','RH','有效顶点','顶点SNR均值','组内相邻parcel对'],
          [[n,len(mapping['mask_sets'][n]),mapping['group_summaries'][n]['lh'],mapping['group_summaries'][n]['rh'],
            stats['groups_and_masks'][n]['n_vertices'],f"{stats['groups_and_masks'][n]['vertex_snr_mean']:.4f}",
            stats['groups_and_masks'][n]['selected_parcel_adjacent_pairs']] for n in names if n!='BASE']),'',
        '随机seed2026092403；每个随机集合在其目标ROI补集内无放回抽取，精确匹配parcel数与LH/RH数量，三个集合彼此不同。SNR、顶点数量、功能和空间邻接只作描述，没有匹配；只有三个集合，不是充分随机化检验。','',
        '## 100图原八项指标','',
        '每候选分别在原100图池评分后平均；BASE直接复用原L120同协议结果。未将两候选组成200图负例池，没有与500图成绩直接相减。Eff/SwAV越低越好，其余越高越好。','',
        table(['条件']+KEYS+['S'],summary_rows),'',
        '## 图像级配对结果','',
        'S=(AlexNet5+Inception+CLIP)/3。D=BASE−遮蔽；E=D_ROI−三个随机集合D的均值。先在每张图内平均两候选，再对100张图配对bootstrap10000次，seed2026092404。两项E的S用名义97.5%区间；其余用95%探索区间。','',
        table(['对比','S差值','区间下限','区间上限','名义水平'],
          [[n,f"{rs[-1]['delta']:.6f}",f"{rs[-1]['low']:.6f}",f"{rs[-1]['high']:.6f}",str(rs[-1]['interval_level']*100)+'%']
           for n,rs in c['contrasts'].items()]),'']
    for roi in ['LL','HL']:
        e=c['contrasts'].get('E_'+roi)
        if e:
            row=e[-1]
            verdict=('在这些固定同大小随机控制下，S额外损失区间位于0以上，支持该固定模型对所定义ROI输入通路有额外依赖。'
                     if row['low']>0 else '在本设计下未确认该ROI相对随机控制的额外S损失；这不等于该脑区没有功能。')
            lines += [f'- {roi}：{verdict}']
        else:lines += [f'- {roi}：匹配控制不可用，未计算E。']
    lines+=['',
        '所有八项的逐图差值和区间保留在comparison.json与paired-*.csv中；对Eff/SwAV，BASE−mask为负表示遮蔽后的距离更高，方向与S相反。不能只依据S或只展示改善。LL/HL数量不同，原始D不可解释为单位脑区重要性。','',
        '## 全部固定图册与AI审阅','',
        '本段图像审阅需由AI实际打开全部固定图册后填写；数值完成标记不等于视觉审阅已完成。最终审阅记录见[AI_REVIEW.md](AI_REVIEW.md)。','',
        '- [所有干预条件并排：GT/BASE/LL/HL/六随机集合，全部原32图双候选](all-condition-atlas/README.md)',
        '- [全部64条IBBI：B_t、步1/10/20/30/40/50 RAM及全时程平均](attention/README.md)','',
        '## 实现与解释边界','',
        '复用原decode_brain_adapter.py生成循环、IPAttnProcessor2_0.ip_attn_map和原八指标。hook返回None，只读真实CFG条件行，并验证CPU/CUDA随机状态不变。每层query维的parcel和约为1；B_t按query数加权层，RAM逐层先L1后跨层等权，再平均已归一化的50步。实际层头数和空间维见attention/manifest.json。200个已选parcel之外的皮层是未评估，不是注意力为0。','',
        '定量512x512 RAM原数组留在受限服务器的manifest所列完整路径，并保存SHA；仓库发布每轨迹未作显示缩放的128x128概率质量块和完整B_t/逐层B摘要。可视化沿用作者Gaussian/对比增强，仅供显示，不替代定量值。','',
        '遮蔽发生于GuidanceGenerator输出E之后、ImageProjModel之前。选中行置零，其他行逐值保持，原无条件分支不变。ImageProjModel偏置/LayerNorm仍在，E置零不等于删除最终K/V或整个通路。这是附录P主题下的明确操作化，未声称Table14逐实现完全复现。','',
        '本模型和数据已反复开发使用；区间是固定模型、固定mask下的描述性名义区间，没有校正历史选择，不是独立test、跨训练seed或人脑因果证据。没有根据注意力/显著性更换权重，也没有扩增样本或另加干预条件。','',
        '依据：[论文§4.1–4.3、附录N/P](https://arxiv.org/html/2509.23566v2)；[官方NSD映射实现](https://github.com/cvnlab/nsdcode/tree/bfd36a503cc90a3eb3ebfd69b269403f7e186924)。','',
        '## 实际预算与停止','',
        f"已完成{complete['ibbi_trajectories']}条IBBI轨迹；新增干预{complete['intervention_images']}个图像条件、{complete['intervention_candidates']}张候选；新增训练更新0。{complete['protected_files']}个受保护历史文件哈希全部一致。原权重和主线best不动。", '',
        '最终控制器退出、GPU释放及三端同步结果将在delivery-checks.json中记录。完成本轮发布后停止，不自动扩大实验。','']
    (REPORT/'README.md').write_text('\n'.join(lines))
if __name__=='__main__':main()

