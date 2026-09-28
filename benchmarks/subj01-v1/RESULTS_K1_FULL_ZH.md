# NSD Subject1 统一K1完整982图结果

五种方法已完成实际重建和同一完整982图八指标评价。标准测试集从官方ID实际核对锁定，并非内部500验证集；每图K=1无选图。各方法原生完整通路、训练范围及采样保留，因此不是等训练预算或等脑输入消融比较。

| 方法 | PixCorr ↑ | SSIM ↑ | AN2 ↑ | AN5 ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NeuroAdapter | 0.071610 | 0.306504 | 72.390075 | 83.442744 | 81.472208 | 83.443990 | 0.804701 | 0.488233 |
| MindEye1 | 0.365033 | 0.288355 | 96.555325 | 98.301330 | 94.264238 | 94.307629 | 0.637356 | 0.357735 |
| Brain-Diffuser | 0.344484 | 0.372837 | 94.688179 | 92.114223 | 72.628516 | 71.268355 | 0.932979 | 0.626845 |
| Takagi CVPR2023 | 0.279962 | 0.223739 | 78.060128 | 76.558481 | 66.985972 | 69.996533 | 0.944405 | 0.674860 |
| MindBridge(s) | 0.157409 | 0.252720 | 85.617050 | 95.646717 | 91.357690 | 91.128696 | 0.761707 | 0.438395 |

MindEye1在七项指标最好，Brain-Diffuser的SSIM最高。MindBridge相对固定NeuroAdapter七项更好、SSIM较低；MindEye1八项均优于MindBridge。Takagi相对NeuroAdapter仅PixCorr及AN2更好；其纹理化和主体失败明显，流程完成不代表论文质量复现成功。

## 图册与可追溯记录

- [NeuroAdapter与完整MindEye1](RESULTS_ZH.md)：原固定32图与AI审阅保留。
- [Brain-Diffuser完整通路](BRAIN_DIFFUSER_test_full_ids_ZH.md)：全八项及固定32图。
- [Takagi原CVPR完整982](TAKAGI_test_full_ids_ZH.md)：原32图、全部失败观察与复现限制。
- [MindBridge(s)600完整982](MINDBRIDGE_test_full_ids_ZH.md)：原32图、逐图观察与恢复/数值设置限制。
- [五种方法及三项缺失的统一CSV](metrics_testfull_single.csv)、[独立首100 CSV](metrics_test100_single.csv)、[模型与回归权重清单](model_manifest.json)。
- [两新增方法的退出码、逐输出SHA、GPU释放及保护审计](takagi-mindbridge-full-audit.json)。
- [Takagi输出manifest](predictions-takagi-testfull.csv)、[MindBridge输出manifest](predictions-mindbridge-testfull.csv)，服务器原输出均保留。

两个新方法固定32图册逐字节不变，复用先前实际完成的非盲AI审阅，未伪称重新盲评。首100表保留且完整982独立全池评分；未将分批识别率平均拼接。Eff使用冻结实现EfficientNet-B1。

## 权重和恢复点

服务器根目录为 /data1/matengyu/geyugong/neuroadapter-author-first，以下路径相对此根。
- NeuroAdapter：runs/subject01-author-fp32-seed42-lr5e-5-from100/exports/epoch-120/inference.pt；SHA256 bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798。
- MindBridge：benchmarks/subj01-v1/upstream/mindbridge/train_logs/benchmark-subj01-vanilla-seed42/last.pth；SHA256 5bd75a992755d70183633df514754b9084b0a4a681284afc83621b2387aac9ad。epoch599即完整600轮、51600更新；同目录last完整双rank恢复状态及loader/best元数据已审计，见mindbridge-training-complete.json。旧421轮快照保留但不替代最终权重。
- Takagi early回归：benchmarks/subj01-v1/processed/takagi-regression/init_latent.joblib；SHA256 2308f46290d1a58318378dc709c555abe6aac2ddc5985cbb4fbbdc34a50c1ec7。
- Takagi ventral回归：benchmarks/subj01-v1/processed/takagi-regression/c.joblib；SHA256 4c98f62d273b49d41a018c054fa8b5bb8c09495728545025e41fe9e78a8195d9。
- Takagi冻结SD1.4：benchmarks/subj01-v1/assets/p1-models/takagi/sd-v1-4.ckpt；SHA256 fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556。
MindEye官方高低层、Brain-Diffuser三回归及生成器的精确来源和SHA见model_manifest.json引用的下载/训练完成记录；不把通用生成器SHA冒充脑回归权重SHA。

## 限制及尚未完成的独立表

MindBridge按原600轮配方训练，但原Accelerate使OneCycle较早到达终点、后半程固定低LR；完整恢复时保留这一行为。推理确定性算法与CUBLAS workspace设置不同于上游默认，已通过原前缀/无GT路径逐元素一致检查，不声称作者历史CUDA数值等价。

Brain-Diffuser保留原全部982预测分布校准；兼容环境、VDVAE随机流等与作者历史环境的差异已披露。Takagi特征抽取使用本次9841图顺序和固定seed，不能声称与作者历史73k分片随机流等价。所有结果为本次K1运行，不完整刻画采样方差，也不是论文多候选表的逐配置复制。本项目历史看过部分test，不是全局盲测，本轮未根据test成绩选择模型/参数/图片。

Cortex2Image缺可下载的官方模型/输入（两Box返回404）；DREAM缺完整原R-PKM脚本与脑侧权重，GIR不能代替；MindFormer(s)未核实可靠原代码/Subject1权重。三项实测均NA，具体来源核验见resource_status.csv与已有资源审计，不断言全互联网绝对不存在。资源索取草稿已准备但未发送。

A表已完成可运行的五种方法。B表作者原生多候选是单独后补工作，尚未执行的行保持NA；MindEye1源码选择器使用脑预测proj_embeddings（不使用GT），可继续审计其原生16候选设置。MindBridge原生8选1依赖GT CLIP，按约束不能执行该选择行；NeuroAdapter匹配brain-encoder选择器未验证。C表没有取得可靠ID的官方输出，不伪填重评分；文献表未填未经逐源核实数值。B/C/文献均不与A表混淆。
