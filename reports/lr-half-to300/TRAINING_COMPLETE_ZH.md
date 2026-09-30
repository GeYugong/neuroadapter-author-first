# L140 至 L300 训练完成与验证启动

2026-09-30 17:37（北京时间），同一低学习率轨迹完成累计 300 轮、159600 次实际更新，skipped_updates=0。训练和控制器均 exit 0，原双卡训练子进程已退出。没有在 200/250 轮暂停评估或修改科学参数。

200/250/300 完整检查点与推理导出已逐文件 SHA 核验；源 checkpoint-140、固定 L120、原指标代码及验证/图册 ID 的保护检查通过。详细证据见 training-final-audit.json。所有恢复状态留在服务器，不上传权重或原数据。

| 轮次 | 推理权重 SHA256 |
|---|---|
| 200 | e1c29913e33534383dbd816699aaee99fd4db775afef27d5ac062ef6bd7b190a |
| 250 | 4f8c3caf1a4795d3dafa60bea5687722f6a5e48d579e525fb915a1e4d67d357c |
| 300 | 91111867b35471f3f63d85c9223584912863023ea5204a10272a453e585461d5 |

## 正在执行的验证

scripts/lr_half_to300_evaluate.py 仅编排已有 evaluate_stage.decode/metrics/atlas 和 lr_half_evaluate.identity/verify_file，依次执行 200、250、300 的完整 val500。每图固定两个候选，50 步、CFG4、seed42、FP32、candidate batch1，不选图；各候选分别在全 500 负例池评分，然后平均。每轮输出独立位于本报告目录 epoch-N 下。既有日志和产物拒绝覆盖。

tmux: author-lr-half-to300-eval；主日志 logs/lr-half-to300-eval-controller.log，各阶段日志 lr-half-to300-eval-N-decode/score.log 和对应真实 .exit。只有全部成功才写 validation-scoring-complete.json；该记录仍明确图册 AI 审阅和模型选择待完成。

目前没有新的验证分数或最终优胜模型。接下来实际审阅固定 32 图，以预定 S 排序比较 L120/200/250/300，精确同分取较早轮次。选优后才开展最终标准 982 K1；若仍选 L120，引用旧结果。旧 best 和基线表保持不变。

验证集已重复用于开发，标准测试已有历史查看。本次是低学习率轨迹延长，不宣称原论文学习率精确复现或独立盲测。训练完成不等于全部验证交付完成。
