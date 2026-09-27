# MindBridge(s) K1推理冻结协议与CPU准备

2026-09-28：训练仍在进行，本记录不代表已取得600轮权重或完成重建。CPU准备脚本实际退出0，未初始化CUDA、未更新训练参数、未生成候选。见[mindbridge-inference-cpu-ready.json](mindbridge-inference-cpu-ready.json)。

## 权重与原生设置

本次只用原作者Subject1 Vanilla/MindSingle、h2048、4blocks、maxpool8192、完整600轮结束后的last.pth。原scripts/inference.sh也使用last；不会按标准测试成绩改用best或其他epoch。600轮完整退出与恢复状态/权重SHA审计完成前禁止启动推理。

原inference.sh设置CFG5、text_image_ratio0.5，options.py默认20步，seed42；生成器为原Versatile Diffusion双引导加UniPC调度器，FP32原始加载默认值。统一K1使用一个候选，不选图。脚本原生候选数为8，和本次K1表分开。

原inference.sh没有指定可选autoencoder_name，因此原生Vanilla入口不使用低层img2img（recon.py设strength=1）。完整保留训练MindSingle产生的图像与文本两条预测条件，不额外拼接MindEye低层网络。此设置依据固定源代码，不宣称已核实作者历史论文每项运行设置。

## 标准测试输入

从既有、SHA已核验的官方Subject1两个test tar中只提取nsdgeneral.npy和coco73k.npy，共982个唯一ID，与锁定标准测试982完全一致，且与8559训练及300原生验证图全部不交叉。没有提取JPG供生成模块使用。

原生输入仍是每图3×15724 float16。严格按原NSDDataset先转float32、对每次重复分别max-pool至8192，再由重建步骤平均重复；不能直接使用MindEye已均值的15724向量再max-pool，因为操作顺序不同。所有982行均通过原数据类加载、维度和有限性检查，逐图原文件SHA及显式ID/原数据集行号映射见[mindbridge-test-inputs.json](mindbridge-test-inputs.json)。sample文件序号没有当作NSD全局ID。

## 冻结生成器CPU检查

在独立MindBridge环境执行原recon.py的prepare_VD，只将UniPC调度器查找重定向为工作区已有官方资产。完整VD在CPU成功加载；16个DualTransformer均为text/image长度77/257、索引1/0、混合比0.5。image_unet与VAE冻结，image_unet全部参数为CPU FP32；不改运行中的训练环境或权重。实际GPU显存、首图接入和正式重建仍待训练结束后核验。

## 原生多候选选择使用GT的核查

固定上游commit 2a3943ff5b619db6f85f16a5e7ff1331d25def55 的src/utils.py::reconstruction在完成生成后，把传入image送入CLIP，再与每个生成候选比较并argmax；recon.py传入的是测试真图。该选择阶段确实依赖GT，不是脑侧预测特征选择。

本次K1无选图，因此后续适配必须截去这一GT选择和绘图部分，生成模块不接收真图；保留此前原生脑条件、随机数、去噪和VAE解码。不能仅因K1的argmax必为0就继续让GT进入生成入口。原生8候选的这个GT选择设置不纳入用户要求的无GT选择B表，明确NA及原因；不临时发明选择器，也不把它与K1实测或文献数值混为一行。

下一步：等待MindBridge完整600轮退出，实际运行该K1路径，与固定NA/MindEye1/Brain-Diffuser/Takagi按同100/982池评价；首100八指标与全部固定32图册先发布。当前没有MindBridge重建分数。
