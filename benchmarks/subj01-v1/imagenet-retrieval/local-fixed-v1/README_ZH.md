# ImageNet 脑响应检索：本地固定协议 v1

**2026-10-06评分更正：** 当前指标与相对比较以[更正报告](scoring-correction-20261006/REPORT_ZH.md)为准。旧baseline-metrics及本目录根部旧CSV/柱状图为错误评分历史记录，已被更正目录取代。完整检索top1及图册未改变。

本地固定协议；encoder开发集划分未核实；标准化及mask与作者历史实验的一致性未确认。

本目录执行用户在 `AUTHORIZATION_ZH.md` 中批准的独立配置。保留父目录严格复现的 BLOCKED/REMEDIATION 历史记录；这些未知项不阻止本轮已授权的探索性执行。新增训练更新为0，不联系作者。

固定使用原982个目标的实测 fsaverage beta：每session每顶点在全部750个trial上对原值/300计算float32均值和总体标准差，按 `(x-mean)/(std+1e-6)` 标准化，再按真实NSD全局0基ID平均3次重复。session统计包含评价trial，不称训练集拟合的标准化。固定经典ROI并集LH16236、RH17380，不含streams。

使用原16成员及每顶点softmax(20×val_corr)权重、冻结DINO、原Schaefer资产。原作者网络严格加载；缓存同批次完全相同的冻结DINO前向，仅计算固定mask所需线性readout，需先与原完整前向通过数值对照。网络为FP32、关闭TF32；拼接33616维响应后用float64计算一次空间Pearson。精确同分按原候选文件名的字典序取较小者。

完整图库1281167个候选，每次编码服务全部982查询；不预筛，不另扫100图，不保存百万张顶点预测。候选清单、逐文件解码/SHA日志、恢复状态及查询数组留在项目runtime。分块状态绑定协议、源码、查询与图库hash。

检索程序不加载GT图片；独立交付程序只在完整扫描完成后评分。检索图保留encoder实际看到的425×425浮点视野；PNG仅供查看。沿用冻结八指标与全982负例池，核对已有方法ID及指标源码指纹后引用L400和四个K1方法。相对变化为描述性比较，保留负数，不作无泄漏泛化或论文Figure5精确复现声明。

当前执行状态以 `run-status.json` 为准；只有完整扫描后才产生 `retrieval-complete.json`，只有评分、图表与图册交付完成后才置 `execution_complete=true`。

[与论文相同GT的24例对照](paper-matched-24/README_ZH.md)：按用户所附论文图完整匹配同ID，新增四页GT／论文baseline／本地baseline图册，既有指标不变。
