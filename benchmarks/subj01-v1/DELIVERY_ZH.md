# NSD Subject1本轮实测交付与剩余缺口

本轮可执行范围已完成；保留无法按原协议执行的行，不启动下一实验。这里的完成指已实际运行的A表五方法及可接入的B表，**不表示八种方法全部完成复现**。

## A表：统一K1

NeuroAdapter固定seed42 L120、完整MindEye1、完整Brain-Diffuser、原CVPR2023 Takagi、单被试MindBridge Vanilla600，均完成同一锁定100与完整982的八指标、输出清单及固定32图册。详见[五方法全982结果](RESULTS_K1_FULL_ZH.md)；原100结果和历史报告保留。MindEye1七项最高，Brain-Diffuser的SSIM最高。训练预算与脑输入不同，不称等预算消融。

## B表：独立原生配置

MindEye1完整高低层原生16候选及脑预测选择已经完成100与982，未使用GT选择；完整982生成、评分均exit0，15712候选全部核验。四页图册与已审阅100版本字节相同，沿用原非盲AI观察。详情：[完整982报告](MINDEYE_NATIVE16_test_full_ids_ZH.md)、[独立100报告](MINDEYE_NATIVE16_test100_ids_ZH.md)、[B表CSV](metrics_native_pipeline.csv)。

|方法/池|PixCorr↑|SSIM↑|AN2↑|AN5↑|Incep↑|CLIP↑|Eff↓|SwAV↓|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|MindEye1原生16 / 982|0.378405|0.290100|97.251651|98.796378|94.651432|94.598284|0.635289|0.352484|

完整982八项均优于A表MindEye1 K1，但A读取PNG、B读取原张量，候选批量的随机流也不同；不能作为纯K消融或宣称严格复现论文数字。Brain-Diffuser原生就是n_samples=1，B的100/982行明确复用A的相同产物和指标，没有新生成。

|剩余原生行|状态与具体原因|
|---|---|
|Takagi原生5候选|NA：官方按五候选相关系数均值做识别，无唯一输出选择器，与本轮冻结八指标不兼容。已读原代码，[审计](TAKAGI_NATIVE_PROTOCOL_AUDIT_ZH.md)。未运行五候选，不拿A首候选冒充。|
|MindBridge原生8候选|NA：原GT CLIP选择违反本轮限制；单被试600训练和K1已完成，不使用多被试权重替代。|
|NeuroAdapter原生8候选|NA：匹配作者brain-encoder选择资产与预处理未验证；不自写选择器。|

## 资源缺口与未实测表

Cortex2Image两官方Box资产链接404；DREAM缺原R-PKM完整脚本及脑侧权重，GIR生成器不能替代；MindFormer未取得可核实的原代码与Subject1权重。检索证据见[资源状态](resource_status.csv)，不宣称全网绝对不存在。上述三项A/B实测均NA。

C表未取得可靠ID和协议的官方逐图输出；文献表未填未经逐源核验数值，不作为本地实测。索取代码、脑侧权重、预处理及ID的[请求草稿](RESOURCE_REQUEST_DRAFTS.md)已准备，未发送邮件或issue。取得原资产后方可另行安排，不自动开启后续实验。

## 保留的复现限制

MindBridge原OneCycle提前到达终点、epoch420保存故障后精确恢复、确定性推理数值环境差异均保留披露。Takagi纹理化和主体失败、Brain-Diffuser模糊、历史test曾被查看等限制不因流程完成而消失。本轮未按test调参、挑权重或重选图片。NeuroAdapter L120及metadata、parcels、best历史SHA复核不变。原八指标代码未更换，100/982负例池独立，未拼接分批识别率。

服务器保存全部原始候选、权重、运行日志及退出码；Git仅发布脚本、协议、指标、清单及图册，不发布权重、原脑数据或凭据。最终同步回执另存，确认三端HEAD/tree后关闭本轮heartbeat。
