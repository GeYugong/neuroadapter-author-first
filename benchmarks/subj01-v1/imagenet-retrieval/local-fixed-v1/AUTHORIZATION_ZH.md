# Codex任务：ImageNet脑响应检索——本地固定协议 v1

## 0. 本次目标与适用范围

不联系作者、不发邮件、不发issue、不新增任何模型训练。基于已经通过前向核查的官方Subject1 encoder ensemble和现有完整ImageNet图库，实际完成一版本地固定协议的检索、八指标及比较图。

这是一项明确的新运行授权建议，不是把历史未知事实改为已验证。保留原严格复现的BLOCKED/REMEDIATION记录，另建local-fixed-v1目录。任务完成不以达到论文分数为条件。

必须在报告、CSV元数据和图注披露：
- 使用公开encoder权重，但其实际训练/验证/ensemble拟合ID未核实；不能声称982图对该encoder全部未见。
- 本轮beta处理和检索mask是开跑前固定的本地实现；不是作者确认的历史配方。
- 相对提升仅为本地描述性比较，不作为已确认无泄漏的泛化性能或论文Figure 5精确复现。

## 1. 项目与已有资产

仓库：GeYugong/neuroadapter-author-first。
最近核查commit：6fa62c08ab1fd3b1ceb04090871563057721d65e。
历史服务器根：/data1/matengyu/geyugong/neuroadapter-author-first。
先检查实际HEAD、工作区、现有进程和产物，若已运行则接续/汇报，不重复启动，不reset或覆盖。

读取已有：
- benchmarks/subj01-v1/imagenet-retrieval/REMEDIATION_ZH.md
- benchmarks/subj01-v1/imagenet-retrieval/public-asset-audit.json
- benchmarks/subj01-v1/imagenet-retrieval/official-weight-verification.json
- benchmarks/subj01-v1/imagenet-retrieval/audit_public_encoder.py
- benchmarks/subj01-v1/imagenet-retrieval/research-evidence.json

复用已核实的16个模型及16个val_corr文件、官方冻结DINO、作者网络类、公开encoder Schaefer资产。不得用NeuroAdapter的200-token映射替换encoder的资产；不得因unknown split重训或更换encoder。

现有候选图库记录为1,261,125张彩色加20,042张灰度，共1,281,167张。复用合法只读资产；灰度转RGB，不漏掉这一部分，不重复下载。

## 2. 一次固定三项定义，不继续等待作者

### 2.1 评价图片

直接使用benchmarks/subj01-v1/test_full_ids.txt里的现有982个目标，以NSD全局0基ID对齐实测fMRI。
这是“重建benchmark的固定评价集合”，不是新证明的encoder held-out集合。
记录：encoder_split_verified=false；encoder_development_overlap=unknown。
不得把未知行随机剔除、截取515行，或通过预测好坏筛选图片。已知且可证实的开发集重叠不能藏在unknown里；若新证据出现，独立披露，不静默改变这版结果。

### 2.2 实测beta处理

本地固定配方参考BrainDiVE@1a42c6e2e2b534b8a28ef2a968fd6ffe1ea4cd37/data_maker/Z_score.ipynb中的load_betas：每session逐顶点标准化，再平均同图重复。
这只是采用其数学处理方式；不使用它的体素空间、train/val划分或图像处理替代本项目fsaverage资产。

对已确认是官方原始存储值的fsaverage beta：
1. 转float32，按参考配方beta/300；若读取的是已经除过300的缓存，避免重复缩放，优先直接读已核实原始数据。
2. 对每个session、每个顶点，在该session全部有效trial上计算mean和std，axis=trial，ddof=0。
3. z=(x-mean)/(std+1e-6)。常数顶点对应z=0，记录数量；对原始NaN/Inf报告真实数据问题，不能无声以nan_to_num掩盖后通过核验。
4. 按真实trial→NSD全局ID映射，对同图可用重复的z取算术平均。记录重复数，不使用排序位置猜ID。
5. 保持左右半球fsaverage顶点身份和顺序一致；只存982查询及必要统计量，不复制整套processed beta。

均值/std使用本session的全部响应，因此包含其中的评价trial。这是已声明的session-wise无标签预处理，不宣称为仅在训练数据上拟合的inductive标准化。图像像素和类别不参与计算。
不比较多种标准化后挑最高分，不用测试GT校准预测响应，不更改NeuroAdapter自身输入或权重。

### 2.3 检索顶点范围

固定使用已经恢复并核实的经典24 ROI/半球并集，不含七个streams。
复用runtime/imagenet-retrieval-assets/public-canonical-roi-union.npz。
预期LH 16236、RH 17380，总33616；将实际掩码SHA和顺序写入本次协议。
不再寻找历史producer、不按结果改范围、不额外开启streams消融。

## 3. 保持原模型计算，明确本地评分定义

16成员全部使用：layer1/3/5/7 × run1/2 × LH/RH。
每顶点ensemble权重沿用原softmax(20*val_corr)；不改简单平均，不重新拟合val_corr，不减少成员。

先得到ensemble预测响应，再抽取本轮固定mask。将LH、RH选中顶点拼接，在拼接后的33616维向量上计算一次空间Pearson：
- 预测向量和实测查询分别减各自空间均值；
- 点积除以两个L2范数乘积；
- 零范数明确报错/记录无效查询，不能赋予任意可用排名。
不得改成各成员相关性平均，或左右半球各算r再随意加权。

图像走已经验证的作者ToTensor→Resize425→CenterCrop425→Normalize及右/下padding到434的路径；沿用已核实的插值、dtype和backend。
检索结果保存实际被encoder观察的425×425视野；padding不进入图像评价。不要另加notebook展示用的256→224裁剪或按结果换构图。

查询来自真实fMRI，而不是目标图像经encoder得到的响应。检索进程不加载NSD目标图像像素或CLIP/DINO目标特征；GT只进入独立评价/图册程序。必要的重复图身份审计单独记录，不作基于语义相似度的候选筛选。

## 4. 只做必要接入检查，然后实际扫库

复用已经通过的权重加载、16成员前向及ROI核验，不重新跑整套资产审计。
新增必要检查限于：
- 5个固定目标的trial、NSD ID、重复数、标准化轴与顶点对应；
- 固定小候选块：流式Pearson与直接float64参考一致；无近并列时top1一致，近并列按固定数值精度和候选ID规则处理；
- 分块顺序及中断恢复后全局top1一致；
- 两张4090上的实际批次吞吐、显存、输出有限性。

工程检查完成就进入本次授权的全库运行；不因三项历史来源未知再次回到等待作者。
真实ID错配、参数加载错误、非有限数值、权限/硬盘不足仍是需要报告的执行故障，不能为了出分而忽略。

一次扫描候选图库，同时对982个查询更新top1。不要为了先跑100个目标而单独再编码一遍百万图库。
两卡可以按左右半球分工；批量大小依据实测显存固定，OOM只能作等价内存优化，不换模型。
不要将所有ImageNet完整顶点预测落盘成TB级缓存；采用候选分块、查询分块、流式精确相关性和global top1。
不做CLIP预筛、近似最近邻、随机缩小图库或每个目标单独编码全图库。

图库读取、图像解码核验和来源hash随实际扫描完成。候选必须有稳定唯一ID；相同分数用预定候选ID顺序决胜。不同目标允许检索到同一候选，不强行一对一分配。

保存可恢复的分片进度与全局top1，绑定模型、查询、图库和协议hash。不得将不同配方/不同精度/不同图库运行的状态混合恢复。
无故障按原授权推进全库，不为例行进度等待新的人工启动。定期记已扫描数和实际速度；ETA来自实际稳定批次，不承诺未经测量的完成小时数。
扫描未完成时只报工程进度；不得把部分图库分数冒充完整1.28M检索结果。
如果发现坏文件，记录候选ID和真实原因，不无声跳过仍宣称完整图库；修复或明确报告有效图库规模。

## 5. 八指标和已有方法比较

全库完成后锁定每个查询的top1候选ID、路径、Pearson分数、图像hash，再进入独立评分。
调用原冻结metric_brain_adapter.py及既有全评价池特征分批方式，只加输出格式适配。
八指标：PixCorr、SSIM、AN2、AN5、Inception、CLIP identification、Eff、SwAV。
保持同982 GT、同预处理和全982负例池。不能分批算识别率再拼均值。

原方法输出和成绩复用，不重新生成：
- NeuroAdapter L400：reports/lr-half-to500/test982/decoded/；metrics/metrics.json；
- benchmarks/subj01-v1下已完成的MindEye1、Brain-Diffuser、Takagi、MindBridge(s) K1。
先核对eval IDs和metric hash，确认匹配再引用。原benchmark里的NeuroAdapter L120不能误标成L400。

新出两张表：
A. 本地检索baseline与各方法八指标原始值；
B. 相对本地检索baseline的描述性变化。
前六项：100*(method-baseline)/baseline。
Eff/SwAV：100*(baseline-method)/baseline。
baseline<=0的相对比例标NA并保留绝对差；不改用任意epsilon制造巨大提升。
保留负提升。图注必须明确“encoder开发集重叠未核实；本地固定预处理/经典ROI”。
旧100k/50图8选1、MindEye16选1和论文引用数字不混入本轮K1主表。

未知开发集暴露和表征配方适配仍可能影响baseline，因此不得把baseline低分直接归因于生成模型更强，也不能把相对柱高当成公平泛化优势。

## 6. 独立交付位置与结束条件

新目录建议：benchmarks/subj01-v1/imagenet-retrieval/local-fixed-v1/。
不要覆盖原严格复现BLOCKED/REMEDIATION、原benchmark主表、L400权重或best。
报告状态分别记录：
execution_complete=true/false；
encoder_split_verified=false；
historical_beta_recipe_verified=false；
historical_retrieval_mask_verified=false；
result_scope=local_fixed_exploratory。

交付协议JSON、候选/查询清单、top1结果、baseline八指标及逐图分数、原始比较CSV、相对比较CSV、可查看柱状图、固定32目标的GT/检索/L400对照和运行记录。
图库和权重不进Git，派生图册遵守已有数据使用权限。图片审阅者类型如实标注；原始测量和限制都保留，不以分数好看作为成功条件。

完成这一个配置后停止。不联系作者，不另起训练，不改标准化/mask追分，不启动其他被试、ROI实验或新的候选选择实验。

## 来源与性质

- 主体检索思路：NeuroAdapter论文第8页§6.1（实测脑响应对ImageNet预测响应进行相关检索）。
- 本地资产核验：当前仓库REMEDIATION_ZH.md及public-asset-audit.json。
- 借用的标准化数学：BrainDiVE@1a42c6e2e2b534b8a28ef2a968fd6ffe1ea4cd37/data_maker/Z_score.ipynb，load_betas函数。
- 982评价池、经典ROI并集选择、已声明未知划分下的探索性运行：本次新拟定的本地协议，不是关于作者历史训练条件的事实声明。
