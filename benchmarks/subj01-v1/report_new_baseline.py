"""Publish separate real K1 tables/atlases; preserve every previous method report."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parent
LABELS = {'neuroadapter': 'NeuroAdapter L120', 'mindeye1': 'MindEye1 full',
          'brain-diffuser': 'Brain-Diffuser full', 'takagi': 'Takagi CVPR', 'mindbridge': 'MindBridge(s) 600'}
KEYS = ['PixCorr', 'SSIM', 'Alex(2)', 'Alex(5)', 'Incep', 'CLIP', 'Eff', 'SwAV']


def picture(method, image_id):
    if method == 'neuroadapter':
        with np.load(B / 'outputs/neuroadapter' / f'image_{image_id:06d}.npz') as z:
            return Image.fromarray(z['candidate_images'][0])
    folder = 'brain-diffuser-full982' if method == 'brain-diffuser' else method
    return Image.open(B / 'outputs' / folder / f'{image_id}.png').convert('RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['takagi', 'mindbridge'], required=True)
    p.add_argument('--pool', choices=['test100_ids', 'test_full_ids'], required=True)
    a = p.parse_args()
    short = 'test100' if a.pool == 'test100_ids' else 'testfull'
    assert (B / f'logs/{a.method}-{short}.exit').read_text().strip() == '0'
    assert (B / f'logs/{a.method}-score-{short}.exit').read_text().strip() == '0'
    methods = ['neuroadapter', 'mindeye1', 'brain-diffuser', a.method]
    metrics = {m: json.loads((B / 'metrics' / m / a.pool / 'metrics.json').read_text()) for m in methods}
    assert len({v['eval_ids_hash'] for v in metrics.values()}) == 1
    assert all(v['K'] == 1 and v['n_eval'] == (100 if short == 'test100' else 982) for v in metrics.values())
    atlas = B / ('comparison_grid_' + a.method)
    atlas.mkdir(exist_ok=True)
    ids = [int(x) for x in (B / 'atlas32_ids.txt').read_text().split()]
    assert len(ids) == 32
    for page in range(4):
        image = Image.new('RGB', (1400, 2276), 'white')
        draw = ImageDraw.Draw(image)
        for col, title in enumerate(['NSD GT'] + [LABELS[m] for m in methods]):
            draw.text((col * 280 + 8, 10), title, fill='black')
        for row, image_id in enumerate(ids[8 * page:8 * (page + 1)]):
            pictures = [Image.open(B / 'assets/gt' / f'{image_id}.png')] + [picture(m, image_id) for m in methods]
            for col, pic in enumerate(pictures):
                image.paste(pic.convert('RGB').resize((256, 256)), (col * 280 + 12, row * 280 + 36))
                draw.text((col * 280 + 12, row * 280 + 294), f'NSD {image_id}; K1', fill='black')
        output = atlas / f'fixed32-page-{page + 1}.jpg'
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        if output.exists():
            assert output.read_bytes() == buffer.getvalue(), 'Existing fixed atlas changed'
        else:
            output.write_bytes(buffer.getvalue())
    lines = [f'# {LABELS[a.method]} 同测试集实测', '',
             f"评价池：{metrics[a.method]['n_eval']}张标准测试图；每图K=1，无选图。",
             '', '| 方法 | ' + ' | '.join(KEYS) + ' |', '|---|' + '---:|' * 8]
    for m in methods:
        lines.append('| ' + LABELS[m] + ' | ' + ' | '.join(f'{metrics[m]["mean"][k]:.6f}' for k in KEYS) + ' |')
    lines += ['', 'Eff/SwAV越低越好，其余越高越好。全部识别指标使用当前同一完整评价池；100图与982图独立评分，不能拼接负例池。', '']
    if a.method == 'takagi':
        lines += ['实际运行原CVPR SD1.4完整脑侧early潜变量与ventral文本回归条件；原50步DDIM日程、strength0.8（实际40步去噪）、CFG5、K1原第一候选。每图seed42在模型初始化前；缓存模型恢复初始化后的随机状态，首个固定目标的冷加载/缓存输出逐像素一致。生成不读取GT条件。', '',
                  '复现限制：兼容环境Python3.10/Torch2.0不等于作者历史环境；上游未锁版本的himalaya使用0.4.11。特征提取固定seed42按本次9841目标顺序，不能声称与作者历史73k分片随机流等价。只训练原脑回归模块，生成器冻结。', '']
    else:
        lines += ['实际使用Subject1 Vanilla完整600轮last，保留图像/文本双脑条件与原VD、UniPC20步、CFG5、mix0.5、seed42。原脚本未指定可选低层AE；未额外拼接MindEye低层。先对每个脑重复maxpool8192再均值，统一评价读取作者原FP32输出张量，PNG仅用于图册。', '',
                  '原训练在421轮保存best发生双rank分歧，完整last审计后继续至600轮；原调度器提前到达终点、后续固定低学习率的实际行为已保留。原生8选1使用GT CLIP，按用户要求列NA，不与无选图K1混用。实际GPU接入因原生重复前向非确定性失败后，明确启用PyTorch确定性算法和CUBLAS_WORKSPACE_CONFIG=:4096:8才通过逐元素身份检查；这不同于上游默认数值运行设置，不声称与作者历史CUDA输出等价。恢复与推理数值限制见MINDBRIDGE_RECOVERY_ZH.md、MINDBRIDGE_INFERENCE_PROTOCOL_ZH.md及实际接入检查记录。', '']
    lines += ['各方法保留原生训练和采样，不代表等训练预算；本项目历史已查看部分测试结果，不是全局盲测。未按本测试成绩调参、选epoch或挑图。旧报告保持不变。', '',
              '以下为预先锁定32目标的全部图册；AI非盲审阅记录单独附后，生成图册本身不代表已完成审阅。', '']
    for i in range(1, 5):
        lines.append(f'![固定32图第{i}页]({atlas.name}/fixed32-page-{i}.jpg)')
    report = B / f'{a.method.upper()}_{a.pool}_ZH.md'
    assert not report.exists(), 'Preserve published report; edit explicitly if needed'
    report.write_text('\n'.join(lines) + '\n')
    with (B / f'{a.method}-{a.pool}-comparison.csv').open('x', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['method', 'n_eval', 'K'] + KEYS)
        for m in methods:
            writer.writerow([m, metrics[m]['n_eval'], 1] + [metrics[m]['mean'][k] for k in KEYS])
    print(report)


if __name__ == '__main__':
    main()
