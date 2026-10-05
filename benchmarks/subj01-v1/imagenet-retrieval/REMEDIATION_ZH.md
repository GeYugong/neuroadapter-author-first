# ImageNet 检索 baseline：深入核查与解除阻塞方案

2026-10-05；基于 323967e 继续。新增训练更新 0。正式全库扫描 0，baseline 八指标尚未产生。本文件补充前一轮 BLOCKED_ZH.md，保留其历史记录，不把诊断前向称为正式检索。

## 这次实际解决的部分

1. **公开权重身份：32/32 文件匹配。** Subject1 的 16 checkpoint 和 16 val_corr 的本地 SHA256，逐一匹配 Hugging Face 官方 revision `d8a978abb212eb2965b5d01673f96536b77e2ea0` 的 LFS SHA。见 official-weight-verification.json。
2. **原模型实际可运行。** 直接导入作者 brain_encoder 原类，只重定向其硬编码 DINO 路径。每个模型用公开 checkpoint 的 25 个键加官方冻结 DINO 的 175 个键构成完整 state_dict；200/200 键严格加载，逐 tensor 回读一致，missing/unexpected 均为空。没有用当前初始化值补缺项，没有改网络或训练。
3. **16 个成员全部完成 GPU 前向。** 固定一张彩色和一张灰度 ImageNet 原图，不加载任何 NSD GT。预处理为作者 ToTensor→Resize425→CenterCrop425→Normalize，再按原 dataset 右/下零填充至 434。原 FP32 模型 batch=2 与逐张前向比较；所有完整顶点输出在任何 nan_to_num 之前均为有限值。预设容差 rtol=atol=2e-4；最大绝对差 9.417533874511719e-6。按原 softmax(20*val_corr) 加权后的 LH/RH 最大差分别 2.7492642402648926e-6 / 2.682209014892578e-6。运行 exit=0，GPU 子进程已退出。
4. **48 个经典 ROI 可准确恢复。** 作者公开 `parcels/nsd_labels/labels_s01.pt` 含 LH/RH 各24组；右半球索引减163842。与官方 NSD native ROI 经官方 white-to-fsaverage 最近邻索引映射的结果，48 个掩码逐个、按原顺序完全一致；与同目录 .npy 也完全一致。并集 LH=16236、RH=17380，总33616顶点。已在项目 runtime/imagenet-retrieval-assets/public-canonical-roi-union.npz 保存派生掩码。它只用于恢复已存在定义，不是开展 ROI 扩展实验，不替换 encoder 的 Schaefer parcels。
5. **完整图库可用。** 上轮已核实共享 train 的1261125张加独立灰度20042张，共1281167张。无需新下载130万图；原图库仍只读。全库解码/逐文件 SHA 及 NSD 重复图审计尚未执行。

诊断代码：audit_public_encoder.py；实际结果：public-asset-audit.json；命令/exit/日志指纹：preflight-run-record.json。以上证明公开模型可以完整加载与运行，不证明内部训练记录和正式检索语义已经全部核实。

## 仍须确认的三个具体事实

### 1. 这16个 checkpoint 的开发集 ID（硬门槛）

当前官方 whole_brain_encoder、其被 NeuroAdapter 固定的历史 commit、同作者 HosseinAdeli/whole_brain_encoder 都直接读取外部 metadata 的 train_img_num/val_img_num/test_img_num。公开 checkpoint 没有这些 ID；现有项目 metadata 是本地重建的，不能作为官方训练记录。README 明确 val_corr 使用 validation split，因而验证集也属于模型开发数据。

早期相关论文 2505.17329 使用 Algonauts 的最后三 session 留出，模型/ensemble 配置也与当前 whole-brain checkpoint 不同。这只能说明划分有不同历史版本，不能据此给当前 checkpoint 指派该划分。NeuroAdapter notebook 的515行同样不足以恢复 ID。

解决办法：取得配套 metadata，或作者确认的三个0基NSD全局ID数组及 val_corr 拟合ID/重复映射；如果所有成员共用划分，一份即可。无需权重重训、原始图片或数十GB beta。按 `原982 ∩ encoder真实held-out IDs − train/validation/权重拟合IDs` 冻结合格集合；逐ID记录排除原因。任何未知 ID 不按“默认无泄漏”放行。不能通过看预测好坏反推或筛选测试集合。

### 2. beta 标准化定义（硬门槛）

编码器论文明确 beta 中心化并缩放为单位方差，但未公开本 checkpoint 的标准化轴、session 范围、ddof、常数列处理以及先标准化还是先平均重复。当前本地 beta 是官方 fsaverage GLMdenoise_RR 原值。

同团队 BrainDiVE 的公开 Z_score.ipynb 提供了一条有来源的候选处理路径：每session逐顶点标准化，之后平均重复；但是该代码使用的是另一模型/空间，不能直接宣称就是当前 checkpoint 的配方。逐顶点缩放会改变跨顶点 Pearson 的排名，不能说“Pearson会自动抵消”。

解决办法：请作者给确切公式/脚本及 derivative/空间说明；若统计量由每session原值确定，现有原始 beta 可分块重建，仅保存合格查询和小型统计量，不必下载或复制完整处理后 beta。若采用训练集统计量，则须拿到统计量或精确训练ID后重算。不要比较几种配方再按八指标选择。

### 3. 历史检索 producer 的 labeled_area 是否包含 streams

经典24 ROI 的数据已完全恢复，但 `fetch.nsd_labeled_area_mask` 会 OR metadata 中所有 ROI 键，而 `img_gen/roi_selection.py` 的另一条 labeled_area 路径显式跳过 early/midventral/midlateral/midparietal/ventral/lateral/parietal 七个 streams。NeuroAdapter notebook 只读取预生成的 ground_truth_sims/labeled_area/corr，公开通用 calc_sims 不能代替这段 producer。

解决办法：确认原 producer 是经典24 ROI并集，还是额外包含七个streams；最好给该 mask 的顶点数量/SHA 或生成代码。若为经典24 ROI，现有掩码可直接使用；若包括streams，则按官方 NSD 同一空间映射补齐，不能凭分数选掩码。此问题是核对原方法定义，不是要求论文内部资产逐字节一致。

## 后续执行顺序（不需要新训练）

- 先收到并核对上述小型来源记录；不要求完整重发 metadata 的所有无关字段。
- 冻结合格 ID、beta 处理、精确顶点掩码和1281167个候选清单。GT 留在独立评分/图册程序。
- 补齐真实 fMRI ID 对应检查，再验证流式 Pearson 与直接参考、稳定并列规则、分块顺序及中断恢复；这部分还没有完成，不能把本轮 batch 前向检查代替它。
- 两张4090可分别承担 LH/RH 的8个原模型，按候选分块，完整ensemble输出后同时更新全部查询的精确top1。每块只临时保留输出；不做CLIP预筛、不缩减成员、不落盘百万完整顶点预测。正式吞吐和耗时需实际批量测量，本轮两图接入测试不能作为全库ETA。
- 全库成功覆盖之后才输出 top1 ID/路径/Pearson/SHA，运行冻结八指标。L400、MindEye1、Brain-Diffuser、Takagi、MindBridge(s)读取已有K1图像；若合格池改变，全部在新池重算识别指标，不重新生成图像。
- 计算本地baseline相对提升，保留负提升；旧100k/50图8选1、MindEye16选1、论文值单列。完成固定图册、真实限制和三端交付后停止。

## 研究边界与结论

本轮已查官方仓库完整可用历史、被固定的旧commit、同作者两个相关仓库、官方HF文件、项目页面、相关论文、BrainDiVE数据准备代码和现有公开issue。未发现这组checkpoint的split或完整beta配方；这是这些已核查入口的结论，不是宣称全网绝对不存在。已停止重复搜索。

前一轮把可恢复的 ROI 资产也直接归入“缺 metadata”的阻塞，排查不够充分；本轮纠正为：模型加载和经典 ROI 资产已验证，尚缺实际划分/标准化来源及原检索 mask 语义确认。upstream_request_draft.md 已备好一个具体的作者询问信草稿，尚未发送；没有代用户发邮件或公开 issue。

未运行训练；未改历史权重、结果、benchmark、best；没有正式baseline数值或相对提升可交付。
