# ImageNet脑响应检索baseline：阻塞报告

状态：BLOCKED，2026-10-05。本轮没有取得实测baseline八指标，没有完成检索。新增训练更新为0，正式检索候选扫描为0，未启动GPU推理。缺项尚未解决，故不生成貌似完成的top1、相对提升表或柱状图。

## 已核实

- 本轮核查基点为author-first@ad2790356f67268f71227b4685b5e8ab43568ff1，初始服务器、本地和GitHub一致；没有同名运行任务。旧训练、输出、benchmark和best不修改。
- 已阅读三个作者notebook、encoder wrapper/fetch/dataset/模型构建及DINO hook、prepare_test.py、evaluate_k1.py、metric_feature_batches.py、metric_brain_adapter.py、已有交付报告和论文相关章节。notebook正式ground_truth_sims/labeled_area/corr与后续model_pred_sims绘图分支不同；后者不得作为本任务查询。
- 16个已有官方checkpoint（layer1/3/5/7×run1/2×lh/rh）均实际CPU反序列化并重新SHA256，验证相关数组也逐个核对，见encoder_manifest.json。每个query_embed为501×768，输出163842顶点。这里只确认资产读取，不冒充全模型strict覆盖、前向一致性已通过。
- 共享ImageNet位于/data1/matengyu/data/ImageNet/ImageNet。train有1,261,125张，train_gray/unpacked_gray另有20,042张；合并无同名重复，总数1,281,167。灰度分片必须包含，不能静默丢弃。train_half和mini不另加作候选。这里只确认文件数量/名称，尚未全库解码与原文件逐SHA，也未确认任何NSD精确图片重复。
- 已有982测试ID的SHA为4c438772ffa978ab8c5f781b8a3fc73df0159ae4892dc39d896f05a413d36441，但它只能证明现有benchmark集合，不能证明encoder无泄漏。
- 检查时双4090空闲，无GPU计算进程；磁盘约108GB可用，原图库只读复用，无需复制整库。

## 实质阻塞：缺少encoder配套metadata及预处理来源

1. **无法冻结合格测试ID。** encoder的datasets/nsd.py直接读取metadata中的train_img_num/val_img_num/test_img_num，main.py用val拟合验证相关，wrapper再用softmax(20*val_corr)逐顶点加权。checkpoint仅有model/epoch/args/val_perf/dataset_type，没有实际训练/验证ID清单；args指向作者内部/engram路径。本地metadata由旧项目prepare_nsd_subj01.py重建，之后本项目另划8500/500；不能当作官方encoder的训练/验证划分。当前982与本项目train/val零交集，不等于与encoder零交集。未将notebook历史515行当作当前ID，未截前515张。
2. **无法确认原labeled_area。** fetch.nsd_labeled_area_mask逐个OR合并metadata的lh_rois/rh_rois。现有metadata只有presentation_order、三种split和lh/rh_ncsnr，缺少ROI字典。不能随意用nsdgeneral、decoder的200parcel或自选ROI代替。
3. **实测beta的encoder输入预处理未确认。** 编码器论文§4.1说明beta中心化并标准化；现有旧项目准备脚本保存原始fsaverage GLMdenoise_RR值。缺少该checkpoint所用标准化轴、统计量和顺序的可核实定义；Pearson不能消除任意逐顶点缩放差异。

官方入口已作有限核查：GitHub whole_brain_encoder当前commit 767f25afc2240f568f4db8c0ce09604e6f83aa72的递归树未提供metadata/split资产；原仓库gitlink 7c7cd1336fe6b68860c412a14a7324a332cdad7d的dataset与当前相同，fetch也依赖上述ROI键。Hugging Face ehwang/brain_encoder_weights revision d8a978abb212eb2965b5d01673f96536b77e2ea0的API列出131个文件，未见metadata、split或parcel文件。服务器HF请求超时后通过本地API成功核查，没有把超时当作资产不存在的证据。未发现可直接补齐上述缺项的官方入口；这不是声称全网绝对不存在。

parcel补充：已有公开schaefer的lh/rh各501parcel，覆盖163842顶点，与checkpoint维度相容；旧审计指出左右membership相同。不能仅据维度宣布身份正确，也不能据此宣布作者一定错误。本轮不要求论文内部逐byte一致，但后续必须验证公开资产自洽、顶点映射和完整加载覆盖。公开checkpoints目录为498/490parcel，不能擅自替换501query的schaefer文件。

## 解除阻塞所需的最小材料

需要与这16个公开Subject1 checkpoint相配套的metadata_sub-01.npy，或等价可核实材料，包含：
- encoder实际训练、验证及ensemble验证权重拟合的NSD全局ID，0/1based与trial/repeat映射；
- 左右半球ROI字典/原labeled_area掩码，fsaverage顶点顺序与来源；
- beta标准化代码/统计量及其与现有NSD derivative的关系；
- 公开schaefer parcel及checkpoint的版本对应说明，或可验证的公开推理资产组合。

未发送邮件或issue。资料齐全后无需重新训练；先冻结合格交集，再做完整加载和小规模等价性检查，通过后按授权直接流式全库扫描。不能用论文baseline数填空，也不能先在不明划分上出正式结果。

## 已锁定的后续评价约束（尚未冻结query集合）

GT只进入评分/图册。检索仅实测fMRI、完整ensemble、原验证相关softmax加权、labeled_area Pearson全库top1。流式候选块同时服务全部query；不缓存百万完整脑预测、不CLIP预筛、不近似搜索。

原notebook的Resize256/CenterCrop224与另一展示的Resize224/CenterCrop224均是绘图代码，未找到正式评分crop证据。若补齐资产后仍无法追溯正式crop，将预先使用encoder空间视野的Resize425/CenterCrop425（不Normalize）作为评价检索图，并明确是本项目统一实现，不按分数选预处理。

评价保留冻结八指标SHA646ee1f3673b033cd6ee485058db16a3d24fae7624d793fe5174008ad2e15f7d及整个合格负例池。L400读取reports/lr-half-to500/test982/decoded，不能把旧benchmark的L120标成L400。MindEye1、Brain-Diffuser、Takagi、MindBridge(s)读取已有K1输出；若集合变化全部重新评分。原100k/50图8选1、MindEye16选1、论文值独立保留，不混入主表。

前六项相对提升100*(模型-baseline)/baseline；Eff/SwAV为100*(baseline-模型)/baseline，负值保留，非正分母NA并保留原始差值。当前baseline缺失，所有相对提升均未计算。固定图册只能使用预定ID，AI非盲审阅不能冒充人工盲测。

## 来源

- https://arxiv.org/html/2509.23566v2 ，§3.4、§6.1、Fig5、附录B/Table2：作为方法/文献参照，不作为本地实测。
- https://github.com/kriegeskorte-lab/whole_brain_encoder
- https://huggingface.co/ehwang/brain_encoder_weights
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12633625/ ，编码器论文§4.1的预处理描述。

本次停止于真实资料门禁。未改变科学参数、未运行训练或其他被试、未覆盖历史表格，也未生成伪造指标。
