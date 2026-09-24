# 八项指标与S：逐图差值、区间和随机对照

全部结果基于同一原100图池，图内先平均两个候选；S单位为识别率百分点，Eff/SwAV为距离且越低越好。D=BASE-mask；E=D_ROI-mean(D_random)。因此Eff/SwAV的负D或负E与其他指标的正值具有相同的损失方向。

## 数据索引

|条件|八项均值/逐候选结果|图内均值|BASE减该条件|
|---|---|---|---|
|BASE|[metrics](../lr-half-to120/val100/metrics.json) · [candidate0](../lr-half-to120/val100/candidate-0.csv) · [candidate1](../lr-half-to120/val100/candidate-1.csv)|[CSV](per-image-BASE.csv)|—|
|LL|[metrics](LL/val100/metrics.json) · [candidate0](LL/val100/candidate-0.csv) · [candidate1](LL/val100/candidate-1.csv)|[CSV](per-image-LL.csv)|[CSV](paired-D_LL.csv)|
|HL|[metrics](HL/val100/metrics.json) · [candidate0](HL/val100/candidate-0.csv) · [candidate1](HL/val100/candidate-1.csv)|[CSV](per-image-HL.csv)|[CSV](paired-D_HL.csv)|
|LL-R1|[metrics](LL-R1/val100/metrics.json) · [candidate0](LL-R1/val100/candidate-0.csv) · [candidate1](LL-R1/val100/candidate-1.csv)|[CSV](per-image-LL-R1.csv)|[CSV](paired-D_LL-R1.csv)|
|HL-R1|[metrics](HL-R1/val100/metrics.json) · [candidate0](HL-R1/val100/candidate-0.csv) · [candidate1](HL-R1/val100/candidate-1.csv)|[CSV](per-image-HL-R1.csv)|[CSV](paired-D_HL-R1.csv)|
|LL-R2|[metrics](LL-R2/val100/metrics.json) · [candidate0](LL-R2/val100/candidate-0.csv) · [candidate1](LL-R2/val100/candidate-1.csv)|[CSV](per-image-LL-R2.csv)|[CSV](paired-D_LL-R2.csv)|
|HL-R2|[metrics](HL-R2/val100/metrics.json) · [candidate0](HL-R2/val100/candidate-0.csv) · [candidate1](HL-R2/val100/candidate-1.csv)|[CSV](per-image-HL-R2.csv)|[CSV](paired-D_HL-R2.csv)|
|LL-R3|[metrics](LL-R3/val100/metrics.json) · [candidate0](LL-R3/val100/candidate-0.csv) · [candidate1](LL-R3/val100/candidate-1.csv)|[CSV](per-image-LL-R3.csv)|[CSV](paired-D_LL-R3.csv)|
|HL-R3|[metrics](HL-R3/val100/metrics.json) · [candidate0](HL-R3/val100/candidate-0.csv) · [candidate1](HL-R3/val100/candidate-1.csv)|[CSV](per-image-HL-R3.csv)|[CSV](paired-D_HL-R3.csv)|

[完整JSON](comparison.json)同时保留每个随机集合和随机均值的全部区间；不是把三组当成独立样本扩增。

## D_LL

[逐图配对CSV](paired-D_LL.csv)

|指标|差值|下限|上限|名义区间|
|---|---:|---:|---:|---:|
|PixCorr|0.002640|-0.005759|0.011630|95%|
|SSIM|0.000710|-0.002855|0.004743|95%|
|Alex(2)|2.156566|0.005051|4.358586|95%|
|Alex(5)|0.929293|-0.742424|2.606061|95%|
|Incep|0.959596|-1.454672|3.454672|95%|
|CLIP|0.828283|-1.303157|3.075758|95%|
|Eff|0.000132|-0.011906|0.012818|95%|
|SwAV|0.004167|-0.003444|0.012130|95%|
|S|0.905724|-0.222264|2.045455|95%|

## D_HL

[逐图配对CSV](paired-D_HL.csv)

|指标|差值|下限|上限|名义区间|
|---|---:|---:|---:|---:|
|PixCorr|0.009433|-0.006009|0.024651|95%|
|SSIM|-0.004802|-0.011209|0.001688|95%|
|Alex(2)|3.818182|0.409091|7.484848|95%|
|Alex(5)|4.510101|1.191919|7.833460|95%|
|Incep|10.191919|5.828283|14.712247|95%|
|CLIP|6.515152|2.489899|10.515278|95%|
|Eff|-0.049825|-0.072197|-0.030192|95%|
|SwAV|-0.035693|-0.051944|-0.021304|95%|
|S|7.072391|4.143098|10.096128|95%|

## E_LL

[逐图配对CSV](paired-E_LL.csv)

|指标|差值|下限|上限|名义区间|
|---|---:|---:|---:|---:|
|PixCorr|0.002606|-0.009812|0.015673|95%|
|SSIM|0.002414|-0.001331|0.006679|95%|
|Alex(2)|1.314815|-0.825042|3.471465|95%|
|Alex(5)|0.348485|-1.185227|1.925926|95%|
|Incep|-1.693603|-4.198653|0.735774|95%|
|CLIP|-0.099327|-3.153241|3.235690|95%|
|Eff|0.011653|-0.002091|0.025698|95%|
|SwAV|0.006648|-0.002629|0.016252|95%|
|S|-0.481481|-2.226150|1.325505|97.5%|

## E_HL

[逐图配对CSV](paired-E_HL.csv)

|指标|差值|下限|上限|名义区间|
|---|---:|---:|---:|---:|
|PixCorr|-0.000856|-0.019635|0.017835|95%|
|SSIM|-0.004804|-0.012893|0.003245|95%|
|Alex(2)|2.361953|-1.639773|6.624705|95%|
|Alex(5)|4.959596|1.718771|8.287963|95%|
|Incep|8.867003|4.296212|13.570960|95%|
|CLIP|5.422559|1.094192|9.845160|95%|
|Eff|-0.038954|-0.063701|-0.016846|95%|
|SwAV|-0.038843|-0.056124|-0.022716|95%|
|S|6.416386|3.022433|9.895665|97.5%|

## 支持与不支持

HL相对三个同大小随机控制：S额外损失6.416386，97.5%名义区间[3.022433,9.895665]。辅助95%区间中AlexNet5、Inception、CLIP及Eff/SwAV支持额外损失；PixCorr、SSIM、AlexNet2的E区间跨零，不支持一概宣称全部指标都受损。SSIM的遮蔽后均值反而高于BASE（0.320885对0.316083），应与语义指标下降一同保留。

LL：S的D=0.905724，95%区间跨零；相对随机的E=-0.481481，97.5%区间[-2.226150,1.325505]。八项E的95%区间也均跨零。AlexNet2原始D的探索区间略高于零，但不足以宣称LL对随机控制有特异的额外效应。Eff/SwAV遮蔽后点估计略低，不能将所有指标概括成退步。

随机控制自身S也不同：LL三个集合83.259259/84.183502/82.365320；HL三个集合83.447811/83.122896/85.430976。HL-R3点估计高于BASE，故不能假设任意同大小遮蔽必然同幅度降低表现。

10000次配对bootstrap，seed2026092404；两项E的S为名义97.5%，其余95%探索区间。固定mask只有三个控制，SNR/顶点/功能/空间邻接未匹配。区间未校正历史自适应选择，内部开发集被反复使用；不是独立测试、人脑因果或脑区无功能的结论。
