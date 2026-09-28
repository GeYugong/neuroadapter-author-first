# 作者原生候选协议B表后补

A表五方法完整982图已发布ecd81a8；B表独立输出与评价，不改变A表，也不按测试质量选择K或参数。

MindEye1原README/src/Reconstructions.py明确16候选。原utils.py第623—634行以脑预测proj_embeddings与候选CLIP比较，非GT选择；高层prior与低层img2img均保留，20步CFG3.5/strength0.85/prior100/seed42。先固定T100首图6489运行原batch16，实测显存与吞吐；失败保留日志，再考虑原实现允许的模块卸载/注意力分片，不降低hidden_dim、不换权重、不凭分数改参数。正式运行前记录实际预算。输出独立outputs/mindeye1-native16，保存全部16原张量与作者selected index，评价用被选原张量；A表MindEye1沿用原PNG评价，这一序列化差异必须披露，不能把它当纯候选数消融。

Brain-Diffuser原脚本n_samples=1，不执行候选择优；已完成的原通路K1结果可作为其原生单候选行复用，明确相同产物，不声称新生成。Takagi原脚本n_iter=5、n_samples=1，尚需核查作者如何汇总而非擅自选best或把首候选称5候选完整结果。

MindBridge原生8选1使用GT CLIP，禁止执行该选择行；NeuroAdapter作者选择器资产及匹配预处理未验证，保持NA，不自写替代选择器。Cortex/DREAM/MindFormer资源阻塞维持；官方输出C表未取得可重评资产。文献表只能填已逐源核实数值，不能填到实测表。

MindEye原batch16接入只按固定首图检查身份、运行和显存，不看效果调参数。全部K1保护仍以protected-before及ecd81a8跟踪文件为准。最终必须分清：A表完成；B表已执行/不适用/阻塞/尚未执行，不能以A表完成伪称附件全部完成。

MindEye native16首100生成与全100负例池八指标评分均exit0，1600原候选逐SHA/形状/范围核验通过；全部固定32图四页已实际AI非盲查看，见MINDEYE_NATIVE16_test100_ids_ZH.md。七项优于K1、SwAV略差，序列化差异已披露，不能称纯K消融。先发布此独立B结果，再原配置复用100推进982；A表不变。

MindEye native16首100已发布3f5d4d7，私有GitHub和四页图册可见核实；真实回执mindeye-native16-test100-publication.json。2026-09-28 10:11+0800在GPU0/tmux bench-mindeye-native16-testfull启动原入口完整982，复用100后生成882，再独立全982负例池评分；启动可用空间约179GiB。查看logs/mindeye1-native16-testfull.log/.exit及mindeye1-native16-scorefull.log/.exit，不重复启动。尚无全982原生16指标。
