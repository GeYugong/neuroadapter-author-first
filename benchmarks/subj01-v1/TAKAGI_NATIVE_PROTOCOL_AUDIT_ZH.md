# Takagi原生五候选协议核查

本次读取固定上游commit e187d4b3db1d647ee3e1b4256a2068ffd15df683的原文件，没有运行新的候选或修改A表。

原codes/diffusion_sd1/diffusion_decoding.py设置n_samples=1、n_iter=5，连续保存000至004五张图；循环不执行脑预测选择器，也不返回唯一选中图。GT在原脚本仅被另存org文件；A适配已经移除这段保存。

原codes/utils/identification.py的load_feat_gen读取五个候选特征。识别时先对同一个GT与五个候选的相关系数取mean，再比较正确图像组与错误图像组：r_true.mean() > r_fake.mean()。因此这是评价阶段的五候选相似度汇总，不是五选一，不是先选最佳，也不是平均五张像素图。

它与本项目冻结的单输出八指标协议不相同。没有可靠作者单图选择规则时，不能自写平均图或best选择器，也不能把A表第一候选标为完整五候选。B表此行明确NA：原生多候选评分无法直接接入冻结八项。原生五候选和原识别率未运行，不冒称完成。A表982个K1真实结果仍有效。

源码定位：[原生成器](https://github.com/yu-takagi/StableDiffusionReconstruction/blob/e187d4b3db1d647ee3e1b4256a2068ffd15df683/codes/diffusion_sd1/diffusion_decoding.py)、[原识别汇总](https://github.com/yu-takagi/StableDiffusionReconstruction/blob/e187d4b3db1d647ee3e1b4256a2068ffd15df683/codes/utils/identification.py)。此结论限于已固定的官方实现，不宣称所有后续版本相同。
