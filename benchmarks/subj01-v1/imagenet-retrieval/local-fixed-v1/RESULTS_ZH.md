# 本地固定协议 v1：完整检索与评价（2026-10-06评分更正）

本地固定协议；encoder开发集划分未核实；标准化及mask与作者历史实验的一致性未确认。

**当前有效指标、比较表和柱状图均在[评分更正报告](scoring-correction-20261006/REPORT_ZH.md)。** 初次四项识别指标因浮点像素略超1触发整批再除255而错误偏低。保持同一批检索图，仅修正数值范围后已完成同982全池重评分。原错误数值文件保留供审计，已作废。

完整1281167候选已扫描；982查询来自实测fMRI；训练新增更新0；GT仅参与检索后的独立评分。检索top1与图册不变。

| 方法 | PixCorr | SSIM | AN2 | AN5 | Incep | CLIP | Eff | SwAV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet local-fixed-v1（更正） | 0.140982 | 0.247521 | 85.197365 | 91.264369 | 75.774024 | 79.433265 | 0.863654 | 0.512661 |
| NeuroAdapter L400 | 0.096383 | 0.277387 | 78.066045 | 88.057512 | 85.078508 | 86.398704 | 0.771142 | 0.455017 |
| MindEye1 K1 | 0.365033 | 0.288355 | 96.555325 | 98.301330 | 94.264238 | 94.307629 | 0.637356 | 0.357735 |
| Brain-Diffuser K1 | 0.344484 | 0.372837 | 94.688179 | 92.114223 | 72.628516 | 71.268355 | 0.932979 | 0.626845 |
| Takagi K1 | 0.279962 | 0.223739 | 78.060128 | 76.558481 | 66.985972 | 69.996533 | 0.944405 | 0.674860 |
| MindBridge(s) K1 | 0.157409 | 0.252720 | 85.617050 | 95.646717 | 91.357690 | 91.128696 | 0.761707 | 0.438395 |

AN2/AN5/Incep/CLIP单位为%；Eff、SwAV越低越好。L400相对更正baseline：PixCorr -31.63%、SSIM +12.07%、AN2 -8.37%、AN5 -3.51%、Incep +12.28%、CLIP +8.77%、Eff +10.71%、SwAV +11.24%。五项改善，PixCorr、AN2、AN5更差；旧七项改善结论撤回。

交付：[每目标top1](top1.csv)、[更正逐图八指标](scoring-correction-20261006/per-image.csv)、[同982原始指标](scoring-correction-20261006/raw-metrics.csv)、[相对变化](scoring-correction-20261006/relative-change-percent.csv)、[绝对差](scoring-correction-20261006/absolute-difference.csv)、[更正柱状图](scoring-correction-20261006/relative-change.png)、[固定图册](atlas32/manifest.json)、[AI非盲审阅](VISUAL_REVIEW_ZH.md)、[更正审计](scoring-correction-20261006/audit.json)。

相对变化为本地描述性比较。session标准化包含当session评价trial；encoder开发集划分及历史处理一致性未知。仅Subject1，不声称无泄漏泛化或论文精确复现。原100k50图best8、MindEye16选1、旧L120和论文值不混入本轮K1主表。旧严格BLOCKED/REMEDIATION、历史模型、best均保留。
