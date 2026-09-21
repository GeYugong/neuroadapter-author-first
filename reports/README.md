# 阶段反馈

固定100图的5/10/20反馈在epoch-5、epoch-10、epoch-20子目录。每个阶段README直接嵌入32验证图和16训练图图册，每张同时显示GT与两个候选。500图与100图识别负例池不同，不能直接相减。

短检查见dataset_check.json、resume_check.json、unchanged_core.json。smoke-decode仅为程序验证，不是性能结果。
