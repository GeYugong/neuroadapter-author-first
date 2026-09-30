# NeuroAdapter L300 → L500 连续续训与交付

用户已授权执行上一轮提出的单条连续续训方案，并设置每两小时检查。保持既有低学习率轨迹，健康训练不中断，不在400/450轮停下等待批准；训练期间不做评价。

- 来源：`runs/subject01-author-fp32-seed42-lr5e-5-from100/checkpoint-300` 完整 Accelerate 状态，300轮/159600更新/0 skipped。终点累计500轮/266000更新，新增200轮/106400更新。
- 来源推理导出SHA：`91111867b35471f3f63d85c9223584912863023ea5204a10272a453e585461d5`。固定旧L120 SHA `bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798`；旧best、全部旧图册与旧结果保留。
- 科学配置不变：原Subject1 train8500、seed42、200个parcel（每半球100）、原网络/TokenDropout/MinSNR损失、冻结SD1.5、AdamW LR5e-5/weight_decay1e-6、FP32与原TF32后端；双4090 DDP每rank batch4、累积2、global16。不新增种子、ROI、低层网络或损失项。
- 模型、完整AdamW、两rank Python/NumPy/Torch/CUDA RNG、独立loader生成器恢复；沿用原逐张量审计并核验下一轮采样顺序。不重置优化器或学习率。
- 每轮完整保存与回读，滚动保留最近两轮；已有105/110/115/120/130/139/140/200/250/299/300全部永久保留，新增400/450/500永久保留完整状态。本轮仅新增400/450/500的推理导出。
- 故障先查实际栈、确认所有子进程退出，保存日志与不完整产物；仅从最新COMPLETE显式恢复，禁止重放旧300。旧日志/started记录不可盲删覆盖。

训练exit0后真实核验500轮/266000更新/0skipped、三个里程碑完整状态及导出SHA、原保护文件不变；记录training-final-audit.json后才能评价。training-complete.json只由成功结束且核验通过的控制器写出，不代表最终交付。

固定验证比较L300/400/450/500。旧L300原val500结果直接引用`reports/lr-half-to300/epoch-300/val500/metrics.json`，不重跑。新三个模型复用evaluate_stage.py与lr_half_evaluate.py，原val500、每图2固定候选、不选图、50步/CFG4/seed42/candidate batch1；原八指标每个候选在全500负例池评价后取两候选均值。完整披露八指标，并实际AI非盲审阅各固定32图册。

模型选择仅按S=(Alex5+Incep+CLIP)/3排序，精确同分取早轮。450相对400、500相对450两个区间若均不足0.3个百分点且其他指标无实质改善，作为停止堆轮数的预算规则；这不是统计显著性检验。无论结果如何，本次授权到500结束，不自动加轮数或新实验。

完成验证选优后，若选中400/450/500，仅该模型开展独立标准982 K1原协议评价与固定32图册；若仍选中300则引用其已完成的982结果。不得按test再次选模型或修改参数。全部新结果位于本reports目录，不覆盖A/B基线表或旧best。

限制：本次为低LR轨迹延长；验证集被反复用于开发且标准test已历史查看，不声称独立盲测或原论文学习率精确复现。论文300轮与当前作者脚本500轮有口径差异，延长的成效由固定验证判断，不保证指标提高。

入口：`runtime/env/bin/python scripts/lr_half_to500.py --resume /data1/matengyu/geyugong/neuroadapter-author-first/runs/subject01-author-fp32-seed42-lr5e-5-from100/checkpoint-300 --launch`。

训练tmux `author-lr-half-to500`，日志`logs/lr-half-to500-controller.log`、`logs/lr-half-to500-from-300.log`及同前缀.exit。后续显式恢复必须使用最新轮次对应新日志。验证入口`scripts/lr_half_to500_evaluate.py`只能训练完成审计通过后执行。最终测试复用已验证的K1入口和隔离评价路径，选优前不启动。

按上一段训练近20轮每轮约14.7分钟估计，200轮训练约49小时，验证及最终测试后整体约53–55小时，按实际吞吐更新。服务器是权威；只使用runtime/env/bin/python，不改共享环境。代码/中文证据以English conventional commit，经服务器git bundle→本地→私有GitHub，核对三端HEAD/tree和跟踪文件一致；不推权重、原始数据或密钥。

每两小时检查真实进程/GPU/日志/完整checkpoint，健康或无实质变化保持安静；完成、故障、重大变化、需用户处理才通知。最终验证、选优及标准测试交付后删除本次heartbeat并停止。
