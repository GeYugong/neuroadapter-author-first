# MindBridge(s) 第421轮保存故障与完整状态恢复

2026-09-28 01:35 +0800，原作者双rank训练在零基epoch420（完成421轮、36206次真实更新）保存best状态时停滞。02:05 NCCL watchdog报告同一SeqNum1176在rank0为ALLREDUCE、rank1为BROADCAST，两rank自动退出，原启动器真实退出码1。未终止其他任务。

原Trainer.save由每个rank独立计算其验证切片上的best判断，而DeepSpeed.save_state需要全部rank参与。原last已在01:34完整写出，随后rank0进入best保存而另一rank进入下一同步流程。这是分布式保存控制流问题，不是测试分数引起的改配方。

## 保存与恢复

- 原失败日志：logs/mindbridge-training.log，退出码logs/mindbridge-training.exit=1。
- 完整恢复快照：processed/mindbridge-recovery-epoch420，逐文件大小和SHA见[mindbridge-recovery-snapshot.json](mindbridge-recovery-snapshot.json)。包含last模型、双rank优化器、调度器和双rank RNG，以及last.pth/best.pth。
- 原不完整best目录保留为upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42/best-interrupted-epoch420，绝不作为完整恢复点。
- 两rank分别验证全部模型tensor、AdamW状态和FP32分片、调度器数值状态以及Python/NumPy/Torch/CUDA RNG与快照相同；恢复审计见[rank0](mindbridge-resume-audit-rank0.json)、[rank1](mindbridge-resume-audit-rank1.json)。
- 原resume使用已完成的epoch420作为循环起点，会重复该轮；本次明确从421（第422轮）开始，不重放训练更新。
- Accelerate0.19为RandomSampler单独创建默认种子的Generator，原save_state没有保存它。本次从相同初始Generator执行421次原BatchSamplerShard的完整索引迭代，恢复其轮次边界状态；不读取训练图、不前向、不更新参数。两rank生成器SHA一致：ff9125d6f5b0acbdde3df69c1032602513253a9d164db570559fc10ece2cb763。此项是依据固定源码和完成轮数的重建，并非与历史未保存的生成器文件直接比对。
- best.pth与last.pth均为epoch420，全部模型tensor一致。原best阈值未持久化，使用该模型重算原生300图验证（分到各rank），随后恢复全部RNG。rank0原生相似度为1.1183711489041646；rank1为1.111934055884679。今后由rank0的原选择标准做决定，再广播给两rank；不改成全rank平均，不使用标准982测试分数。
- 已通过真实双GPU状态恢复与best重新完整保存，日志出现RESUME_AUDIT_PASSED_BOTH_RANKS并进入epoch421。训练仍为partial，尚未达到600轮，不填入完整实测表。

## 最小工程修改与限制

新增独立恢复入口mindbridge_resume_train.py及防重复启动器launch_mindbridge_resume.py，上游src保持原commit不变。运行时仅替换恢复函数、同步save分支、保留train开头的best历史；原训练前向、损失、优化器、网络、批量、600轮、数据和采样设置不改。今后每次完整保存附带best历史及每rank数据生成器状态。首次恢复的快照固定且不覆盖，启动器拒绝重复启动。

CPU双进程Gloo测试覆盖rank0判断真/rank1假及反向情形，两rank的保存调用序列一致；见[mindbridge-recovery-test.json](mindbridge-recovery-test.json)。实际双GPU审计独立记录，CPU测试不替代GPU验证。

恢复时发现原运行OneCycleLR last_epoch已为51600、_step_count=51601，优化器global_steps=36206。原Accelerate包装器在split_batches=False时每更新推进两次调度，达到OneCycle总步数后停止推进。因此原运行在约300轮后维持学习率2.4000055952305e-07。此为本次固定作者代码/环境组合的实际行为，已如实保留；没有趁恢复重新设学习率或延长scheduler。不能宣称与作者未核实的历史软件环境逐位等价。

恢复验证是状态级验证，不承诺重启后的CUDA数值轨迹逐位等同于未中断运行。没有新增NeuroAdapter训练，没有更换测试ID或按测试成绩选权重。

## 后续入口

tmux bench-mindbridge-resume；logs/mindbridge-resume.log及结束后的mindbridge-resume.exit。只允许先查实际进程、日志、最新完整状态；不能重复运行这次固定epoch420恢复启动器。若再次故障，读取堆栈和最新checkpoint附加的trainer-recovery.json/loader_generator文件后另做明确恢复。

600轮完整结束、确认双GPU退出后，再开展单被试完整推理及Takagi原CVPR重建。同100图K1八指标/32图册先发布，再完整982；旧NeuroAdapter、MindEye1、Brain-Diffuser结果不重跑。

## 完成更新

2026-09-28 07:12完成零基599/600轮并退出0，51600更新/0 skipped；最终导出和双rankFP32分片重建完全一致。见mindbridge-training-complete.json。AdamW内部step为51601，已核实DeepSpeed0.9.5 initialize_optimizer_states在零梯度上调用一次step进行初始化；未把该初始化计作训练批次。模型选择仍为完整600 last，而非验证best590。
