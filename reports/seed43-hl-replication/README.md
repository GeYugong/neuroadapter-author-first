# Seed43固定HL内容复核（训练启动准备）

用户已授权启动，预算沿用[冻结计划](../evidence-freeze-seed43-plan/PLAN.md)。新训练seed43；100轮LR1e-4完整保存并退出后，完整恢复模型/AdamW/双rank RNG/DataLoader，再仅改LR至5e-5，固定120轮结束。预计63840更新，以实际日志核实，不按分数选epoch。不使用seed42权重继续训练；原主线与L支线保持停止。

训练命令（仓库根目录）：

```sh
runtime/env/bin/python scripts/seed43_replication.py
```

控制器串行调用同一个scripts/run_train.py --seed43-replication --target100/120（实际参数分开），仅使用原作者训练循环。端点训练退出后才独立评价；当前控制器只负责训练，评价接入及实际结果尚待完成，不把TRAINING_COMPLETE当作实验完成。

双rank初始化审计验证FP32、参数一致、新seed的guidance不同于seed42初始值、parcel映射与L120精确一致，审计不改变RNG。既有audit记录首个optimizer更新的前后参数。100→120恢复在双rank核对模型/AdamW/RNG/loader，除预定LR外优化器状态不变。中断后确认进程全退出，仅显式--resume本run最新COMPLETE，不重放初始化；代码保留原阶段和LR保护。

新正式推理2000候选（AA200+三供体BA/AB/BB1800），检查最多8，总上限2008。复用原目标/供体/HL/随机采样规则，但seed43的AA必须新生成；全部条件同一100目标GT池评分，24页固定图册，D/R块bootstrap按冻结计划。当前没有新评价结果。

见[实现核验](implementation-check.json)、[入口边界测试及状态](preflight.json)。旧内容复核433项锁及全部既有已跟踪报告哈希作为保护清单；旧best和历史实验不得被新模型替换。
