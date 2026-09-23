# 同起点完整AdamW半学习率短程开发实验（开训前固定）

B0=原100/53200，不更新；H=原110/58520（LR1e-4），不重训；L=独立subject01-author-fp32-seed42-lr5e-5-from100，从完整100恢复后只将LR改为5e-5，固定训练10轮5320更新到110/58520。105仅保存不评分调参。源checkpoint只读，主线暂停记录、历史权重、best及历史报告不动。

复用作者入口与训练循环，双rank完整Accelerate加载后，先逐张量核对模型/AdamW和param_groups，再修改LR并验证除此之外完全一致；不在恢复后重新设seed。恢复L自身时加载其最新完整状态和5e-5，不重放100。正式更新前按复制的batch sampler推演101顺序与H逐epoch哈希核对，实际逐epoch记录再次核对。其他科学配置全部继承主线。

评价：原500内部验证ID，两候选独立生成，batch1/50步/CFG4/原ID seed与完整随机过程/原精度backend/无选择/GT不参与生成。B0复用既有500；H核实原100身份并只补400（800候选），独立补评目录；L生成500及固定训练16。候选分别对完整500池评分后平均，并从相同输出评分monitor100，不能拼接100/400识别率或合成1000负例池。全部结果写本实验独立报告，不写主线报告。

主要比较L110-B0、L110-H；主指标固定S=(Alex(5)+Incep+CLIP)/3。图像内先平均两候选，以图像为单位做10000次配对bootstrap，NumPy default_rng seed20260923，两个主要S对比各报告97.5%百分位区间（1.25%,98.75%）；八指标同时报告，辅助指标95%探索区间，Eff/SwAV越低越好。L同时超过B0和H才支持实际收益；仅超过H不称修复。若点估计改善但区间跨零，报告收益未确认，不说无效或修好。反复使用的内部开发集，不是独立测试或跨seed结论。

实际AI审阅原32验证+16训练全部候选，展示改善和内容退步，不以S替代图册。无额外LR/seed/探针/诊断。只本次10轮和统一评价，结束停止，新候选不自动提升主线best。数值/恢复/顺序不一致时保留最后完整状态停止报告。

运行：`runtime/env/bin/python scripts/lr_half_train.py`。中断后须先确认原控制器和所有子进程结束，仅从L最新COMPLETE恢复：`runtime/env/bin/python scripts/run_train.py --branch-lr-half --target 110 --resume runs/subject01-author-fp32-seed42-lr5e-5-from100/checkpoint-N`，N必须100以上且小于110；不得重新执行初始控制器。恢复审计为每rank独立JSON。

开训前工程修复：采样器deepcopy原先尝试复制HDF5 dataset句柄而失败，未进入训练循环、0新增更新。改为`copy.deepcopy(loader.batch_sampler, {id(dataset): dataset})`，dataset仅共享只读引用；不改数据顺序与科学设置。确认控制器/启动器/双rank全部退出且L输出目录为空后，允许这次零更新重试。
