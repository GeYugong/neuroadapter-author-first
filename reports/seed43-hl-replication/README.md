# Seed43固定HL内容复核（正在训练）

用户已授权启动，预算沿用[冻结计划](../evidence-freeze-seed43-plan/PLAN.md)。新训练seed43；100轮LR1e-4完整保存并退出后，完整恢复模型/AdamW/双rank RNG/DataLoader，再仅改LR至5e-5，固定120轮结束。预计63840更新，以实际日志核实，不按分数选epoch。不使用seed42权重继续训练；原主线与L支线保持停止。

训练命令（仓库根目录）：

```sh
runtime/env/bin/python scripts/seed43_replication.py
```

控制器串行调用同一个scripts/run_train.py --seed43-replication --target100/120（实际参数分开），仅使用原作者训练循环。端点训练退出后才独立评价；当前控制器只负责训练，评价接入及实际结果尚待完成，不把TRAINING_COMPLETE当作实验完成。

双rank初始化审计验证FP32、参数一致、新seed的guidance不同于seed42初始值、parcel映射与L120精确一致，审计不改变RNG。既有audit记录首个optimizer更新的前后参数。100→120恢复在双rank核对模型/AdamW/RNG/loader，除预定LR外优化器状态不变。中断后确认进程全退出，仅显式--resume本run最新COMPLETE，不重放初始化；代码保留原阶段和LR保护。

新正式推理2000候选（AA200+三供体BA/AB/BB1800），检查最多8，总上限2008。复用原目标/供体/HL/随机采样规则，但seed43的AA必须新生成；全部条件同一100目标GT池评分，24页固定图册，D/R块bootstrap按冻结计划。当前没有新评价结果。

见[实现核验](implementation-check.json)、[入口边界测试及状态](preflight.json)。旧内容复核433项锁及全部既有已跟踪报告哈希作为保护清单；旧best和历史实验不得被新模型替换。

2026-09-25 01:19:21 +0800：tmux author-seed43-replication启动，控制器1764367、启动器1764370。双rank初始化审计通过，真实第1轮（零基Epoch0）已运行，首个optimizer更新的参数变化已核验。科学代码提交64df93e；尚未完成训练或评价。详情见STARTED.json、initial-rank0/1.json和first-update.json。

## 独立评价入口已接入（尚未运行）

训练控制器完成并退出、GPU空闲后，运行 `runtime/env/bin/python scripts/seed43_evaluate.py`。入口先核验120完整状态及100/110检查点，分独立进程执行prepare、identity、baseline、generate、score、direction、compare、atlas、finish。复用既有三供体评分/统计与原生成器，不自动改best。

接入检查固定上一轮新目标清单的第1图及供体批次1：原decoder脑输入路径与内容包装器各生成AA/BB两个候选，总8张；逐像素、GT和seed比较通过后才生成2000正式候选。新AA200张独立生成；历史seed42文件仅为目标/供体GT来源和旧结果对照，不冒充新AA。正式复核输出/evaluation报告独立，24页固定图册；总预算2008不变。

CPU入口、错误路径和检查预算测试见evaluation-wiring-check.json。训练尚未完成，模型加载和像素恒等验证尚待真实120权重；不能将这些静态检查写成推理已通过。
