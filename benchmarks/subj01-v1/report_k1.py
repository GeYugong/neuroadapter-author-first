from pathlib import Path
import csv,json,hashlib,argparse
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--ids',type=Path,required=True);a=p.parse_args()
models=json.loads((B/'model_manifest.json').read_text())
ids=[int(x) for x in a.ids.read_text().split()]
columns=['method','variant','subject','train_subjects','n_train_images','n_eval','eval_ids_hash','K','selection','result_source','PixCorr','SSIM','AN2','AN5','Incep','CLIP','Eff','SwAV','model_sha','code_sha','notes']
rows=[]
for method,key in [('NeuroAdapter','neuroadapter'),('MindEye1','mindeye1')]:
 f=B/'metrics'/key/a.ids.stem/'metrics.json'
 if not f.exists():raise RuntimeError('First result requires both actual methods: '+str(f))
 data=json.loads(f.read_text());model=models[method];mean=data['mean']
 print(method,mean)
 aliases={'PixCorr':['PixCorr'],'SSIM':['SSIM'],'AN2':['Alex(2)','AlexNet(2)','AlexNet_2','AN2','AlexNet2'],'AN5':['Alex(5)','AlexNet(5)','AlexNet_5','AN5','AlexNet5'],'Incep':['Inception','Incep'],'CLIP':['CLIP'],'Eff':['EffNet','EfficientNet','Eff'],'SwAV':['SwAV']}
 row=dict(method=method,variant=model['variant'],subject=1,train_subjects='1',n_train_images=model['n_train_images'],n_eval=len(ids),eval_ids_hash=data['eval_ids_hash'],K=1,selection='none',result_source=model['result_source'],model_sha=model.get('weight_sha256',json.dumps(model.get('weights',{}),sort_keys=True)),code_sha=model.get('upstream_commit',json.loads((B/'startup.json').read_text()).get('git_head','see startup.json')),notes='Native complete pathway; fixed K1; no test-based selection')
 for name,keys in aliases.items():
  found=[k for k in keys if k in mean]
  assert len(found)==1,(name,mean)
  row[name]=mean[found[0]]
 rows.append(row)
for method in ['Cortex2Image','Brain-Diffuser','Takagi CVPR2023','MindBridge(s)','DREAM','MindFormer(s)']:
 row={k:'NA' for k in columns};row.update(method=method,subject=1,n_eval=len(ids),K=1,selection='none',result_source='blocked',notes='No completed local result yet; see resource_status.csv');rows.append(row)
name='metrics_test100_single.csv' if a.ids.name=='test100_ids.txt' else 'metrics_testfull_single.csv'
with (B/name).open('w') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
atlas=[int(x) for x in (B/'atlas32_ids.txt').read_text().split()]
out=B/'comparison_grid';out.mkdir(exist_ok=True)
for page in range(4):
 canvas=Image.new('RGB',(3*280,8*280+36),'white');d=ImageDraw.Draw(canvas)
 for col,title in enumerate(['NSD GT','NeuroAdapter L120 K1','MindEye1 full K1']):d.text((col*280+8,10),title,fill='black')
 for row,i in enumerate(atlas[page*8:(page+1)*8]):
  with np.load(B/'outputs/neuroadapter'/f'image_{i:06d}.npz') as z:na=Image.fromarray(z['candidate_images'][0])
  images=[Image.open(B/'assets/gt'/f'{i}.png').convert('RGB'),na,Image.open(B/'outputs/mindeye1'/f'{i}.png').convert('RGB')]
  for col,im in enumerate(images):
   x=col*280;y=36+row*280;canvas.paste(im.resize((256,256)),(x+12,y));d.text((x+12,y+258),f'NSD {i}; candidate 0',fill='black')
 canvas.save(out/f'fixed32-page-{page+1}.jpg',quality=95)
lines=['# NSD Subject1 同测试集重建比较','',f'本表评价池：{len(ids)} 张；K=1，无选图。测试清单 SHA256：`{rows[0]["eval_ids_hash"]}`。','', '两种方法保留各自完整脑侧通路、原生预处理和采样。NeuroAdapter 为本项目固定 L120 复现权重，MindEye1 为官方 Subject1 高层与低层权重。该比较不代表等训练预算或论文原生多候选协议。','', '| 方法 | PixCorr ↑ | SSIM ↑ | AN2 ↑ | AN5 ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows[:2]:lines.append('| '+row['method']+' | '+' | '.join(f'{row[k]:.6f}' for k in ['PixCorr','SSIM','AN2','AN5','Incep','CLIP','Eff','SwAV'])+' |')
lines+=['','Eff 使用冻结实现的 EfficientNet-B1。所有识别指标在同一完整评价池计算；100 图预览与完整测试集分表，不能加权拼接。项目历史看过部分标准测试结果，本轮不根据分数改变模型、参数或候选。','','## 固定32图图册','']
for n in range(1,5):lines.append(f'![固定32图第{n}页](comparison_grid/fixed32-page-{n}.jpg)')
lines+=['','其他方法状态见 [resource_status.csv](resource_status.csv)。未完成方法没有实测分数；Cortex2Image 官方资产链接当前返回404，不能用论文拼图代替。','', '逐图分数保存在 `metrics/<method>/<pool>/per-image.csv`；原始输出、模型和随机协议独立保存。']
(B/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
print('REPORT',name)
