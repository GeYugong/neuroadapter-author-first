from pathlib import Path
import json,csv,argparse
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--pool',choices=['test100_ids','test_full_ids'],required=True);a=p.parse_args()
methods=['neuroadapter','mindeye1','brain-diffuser'];names=['NeuroAdapter L120','MindEye1 full','Brain-Diffuser full']
metrics={m:json.loads((B/'metrics'/m/a.pool/'metrics.json').read_text()) for m in methods}
assert len({v['eval_ids_hash'] for v in metrics.values()})==1
out=B/'comparison_grid_bd';out.mkdir(exist_ok=True)
ids=[int(x) for x in (B/'atlas32_ids.txt').read_text().split()]
for page in range(4):
 im=Image.new('RGB',(1120,2276),'white');d=ImageDraw.Draw(im)
 for c,title in enumerate(['NSD GT']+names):d.text((c*280+8,10),title,fill='black')
 for row,i in enumerate(ids[8*page:8*(page+1)]):
  with np.load(B/'outputs/neuroadapter'/f'image_{i:06d}.npz') as z:na=Image.fromarray(z['candidate_images'][0])
  pics=[Image.open(B/'assets/gt'/f'{i}.png'),na,Image.open(B/'outputs/mindeye1'/f'{i}.png'),Image.open(B/'outputs/brain-diffuser-full982'/f'{i}.png')]
  for c,pic in enumerate(pics):im.paste(pic.convert('RGB').resize((256,256)),(c*280+12,row*280+36));d.text((c*280+12,row*280+294),f'NSD {i}; K1',fill='black')
 im.save(out/f'fixed32-page-{page+1}.jpg',quality=95)
keys=['PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV']
lines=['# Brain-Diffuser 完整通路实测','',f"评价池：{metrics[methods[0]]['n_eval']}张标准测试图；每图K=1，无选图。",'','| 方法 | '+' | '.join(keys)+' |','|---|'+'---:|'*8]
for m,n in zip(methods,names):lines.append('| '+n+' | '+' | '.join(f'{metrics[m]["mean"][k]:.6f}' for k in keys)+' |')
lines+=['','Eff/SwAV越低越好，其余越高越好。三个脑回归和VDVAE低层＋完整VD双条件通路均实际执行，非低层中间图分数。Brain-Diffuser保留50步日程、strength0.75（实际37步去噪）、CFG7.5、mixing0.4、作者一次seed0原982行序，100图从该输出提取，不重排随机流。','','复现披露：兼容环境Python3.10/Torch2.0与作者旧版本不同；VDVAE原脚本未实际设置H.seed，本次固定0；原仅取形状的GT预热改用同shape全零输入，两种batch的随机终态检查通过，生成不读取GT。不能声称与作者历史输出逐像素相同。所有方法未依据本测试成绩调参。','','首100与完整982独立评分，识别率不拼接；旧P0报告保留。固定32图全部候选如下，AI审阅记录另附。','']
for i in range(1,5):lines.append(f'![固定32图第{i}页](comparison_grid_bd/fixed32-page-{i}.jpg)')
(B/f'BRAIN_DIFFUSER_{a.pool}_ZH.md').write_text('\n'.join(lines)+'\n')