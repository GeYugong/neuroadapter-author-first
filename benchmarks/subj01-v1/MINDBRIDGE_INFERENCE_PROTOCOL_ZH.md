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

## K1生成函数适配与CPU差分验证

mindbridge_k1_native.py已从固定SHA的原utils.reconstruction直接编译生成部分。函数签名仅接收脑向量、脑模型、UNet、VAE、scheduler和device，无GT或CLIP选择参数；在原best_picks之前截断，保留原两路脑预测条件归一化、CFG、私有seed42生成器、20步去噪和VAE解码。对原Vanilla已固定的mem_efficient=False和无可选AE分支做静态展开，没有新增替代生成算法。

CPU测试实际退出0：用轻量合成模块，将适配函数输出与未经改写的原函数生成前缀作逐元素比较；参考前缀分别传全零/全一合成占位图，两次均完全一致，CLIP编码函数一旦被调用会直接报错。两路预测条件均进入334-token条件，20次去噪调用、全局CPU RNG不变和重复调用一致均已验证，CUDA未初始化。见[mindbridge-k1-adapter-cpu-test.json](mindbridge-k1-adapter-cpu-test.json)。合成模块只验证代码适配，不是实际模型结果，也不替代后续GPU接入检查。

当前只有生成函数适配，不是可启动完整评价的控制器。后续仍须在600轮完整退出后检查最终last及双rank恢复状态，接入实际模型和锁定ID，首个固定目标比较原生成前缀与适配路径的实际GPU输出，保留候选记录与SHA、断点和退出码，再评分/图册。不得将本CPU测试写成MindBridge重建完成。

## 完整推理入口预检（仍未启动GPU）

mindbridge_reconstruct_k1.py及launch-mindbridge-test100.sh/testfull.sh已接好原模型/ID/逐图记录与统一评分。默认仅CPU预检：100/982的原始脑重复、文件SHA、ID映射、上游源码及无GT生成函数均已通过。实际--generate在当前训练未完成时明确拒绝，未创建输出或加载GPU模型；不是训练故障，见mindbridge-decoder-guard-test.json。

入口必须读取真实的mindbridge-training-complete.json，要求completed_epochs=600、global_steps=51600、resume_exit_code=0、full_state_verified=true及最终last.pth的checkpoint_sha256；这份记录尚不存在，必须训练完全退出并实际检查完整双rank状态后才填写，不能只为通过检查伪造。最新last的trainer-recovery.json也必须为599/next600，并要求指定GPU无计算进程。严格加载完整MindSingle权重，不改用best。

首个固定目标用实际完整模型比较原生成前缀（仅合成零占位图）与无GT适配函数的FP32输出，完全相同才接收正式结果；多一次身份检查不参与挑图。正式每图只保留一个候选，保留作者原始FP32张量给统一评价，PNG只供图册，避免额外uint8量化改变原生输出。逐图保存脑输入、模型、代码及输出SHA。发生故障保留所有记录和孤立产物，显式--resume会校验已完成图，禁止盲覆盖。

完整982还要求首100先真实评分、全部32图册审阅和发布；mindbridge-test100-publication.json须记录n_eval=100、atlas_images_reviewed=32、同一checkpoint_sha256和真实published_commit。当前这些结果均未产生，实际GPU接入、显存及重建分数仍未验证。
