# L400 低学习率平衡续训：执行记录

用户已授权执行 `reports/next-training-plan-20261007/PLAN_ZH.md`，并将监控改为每小时一次。此文件记录执行边界，不代表训练已开始或实验已完成。

- 独立轨迹：`subject01-author-fp32-seed42-lr1e-5-from400`。完整恢复旧 L400 的模型、AdamW、双 rank RNG 和 loader，只将学习率从 5e-5 改为 1e-5。
- 先完成旧 L400、旧 L500 的固定噪声诊断；每个权重最多两小时、零训练更新。通过严格加载、parcel、查询清单检查后等待两张 GPU 空闲。其他任务不被停止或抢占。
- 新轨迹累计 425/450/475/500 为保存后诊断边界。450 完成固定 val500 K2 后按既定门槛决定是否继续；最多新增 100 轮。仅导出450/500。425/475使用完整checkpoint诊断。
- 门槛对比旧 L400：S/PixCorr/SSIM均不下降，且 S 至少提高0.3pp或PixCorr与SSIM均提高0.005；四个识别指标分别最多下降0.5pp，Eff/SwAV分别最多上升0.005。此为预算规则，不是显著性检验。
- 完成后仍需查看固定图册、选择最终模型、按计划处理标准982、完成保护核验和报告同步；控制器只标记 `validation_complete_review_pending`，绝不据此宣称交付完成。

入口：`scripts/balanced_controller.py`；训练入口复用原实现，仅增加完整恢复、日志和导出边界：`scripts/balanced_train.py`、`balanced_resume.py`。验证复用原冻结解码/评分函数：`scripts/balanced_evaluate.py`。

已通过静态编译与导入、源checkpoint/配置核验、门槛正例与退化否决测试、TokenDropout输出与全局RNG不变测试。尚不能用这些测试代替真实双rank恢复和首个更新审计。实际阶段以 `controller-status.json`、进程、日志、checkpoint为准。

发生失败时保留所有日志和部分产物，确认子进程退出后再处理；禁止重复启动、覆盖启动记录或回到旧L400重跑已完成更新。恢复需从该新轨迹最新COMPLETE显式执行，并保留新attempt记录。

源L400完整状态及旧报告/benchmark/配置受SHA保护。训练数据、网络、MinSNR、TokenDropout、FP32/原TF32后端均保持原方案。验证反复使用、标准测试已有历史查看；不声称独立盲测。
