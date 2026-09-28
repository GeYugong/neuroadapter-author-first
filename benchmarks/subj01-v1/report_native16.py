"""Independent native16 report; validates stored candidates and never regenerates."""
from pathlib import Path
import argparse,csv,hashlib,json,datetime,io
import torch
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--pool",choices=["test100_ids","test_full_ids"],required=True);a=p.parse_args()
 short="100" if a.pool=="test100_ids" else "full"
 assert (B/f"logs/mindeye1-native16-test{short}.exit").read_text().strip()=="0"
 assert (B/f"logs/mindeye1-native16-score{short}.exit").read_text().strip()=="0"
 ids=list(map(int,(B/(a.pool+".txt")).read_text().split()))
 met=json.loads((B/"metrics/mindeye1-native16"/a.pool/"metrics.json").read_text())
 assert met["K"]==16 and met["n_eval"]==len(ids) and met["eval_ids_hash"]==sha(B/(a.pool+".txt"))
 out=B/"outputs/mindeye1-native16"
 torch.set_num_threads(4)
 rows=[]
 for i in ids:
  r=json.loads((out/f"{i}.json").read_text())
  assert r["nsd_id0"]==i and r["K"]==16 and 0<=r["candidate"]<16
  cp=out/r["candidates_path"]
  assert sha(cp)==r["candidates_sha256"] and sha(out/f"{i}.png")==r["prediction_sha256"]
  t=torch.load(cp,map_location="cpu",weights_only=True)
  assert t.shape==(1,16,3,512,512) and torch.isfinite(t).all() and t.min()>=0 and t.max()<=1
  rows.append(dict(nsd_id0=i,K=16,selected_index=r["candidate"],candidate_path=str(cp.resolve()),candidate_sha256=r["candidates_sha256"],selected_png_sha256=r["prediction_sha256"],native_dtype=str(t.dtype),selection="original brain-predicted CLIP selector; no GT"))
 with (B/f"mindeye-native16-{a.pool}-manifest.csv").open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");w.writeheader();w.writerows(rows)
 atlas=B/"comparison_grid_mindeye_native16";atlas.mkdir(exist_ok=True)
 fixed=list(map(int,(B/"atlas32_ids.txt").read_text().split()))
 for pg in range(4):
  im=Image.new("RGB",(840,2276),"white");d=ImageDraw.Draw(im)
  for col,label in enumerate(["NSD GT","MindEye1 K1 (A)","MindEye1 native16 (B)"]):d.text((col*280+8,10),label,fill="black")
  for row,i in enumerate(fixed[pg*8:pg*8+8]):
   r=json.loads((out/f"{i}.json").read_text())
   for col,path in enumerate([B/"assets/gt"/f"{i}.png",B/"outputs/mindeye1"/f"{i}.png",out/f"{i}.png"]):
    pic=Image.open(path).convert("RGB");im.paste(pic.resize((256,256)),(col*280+12,row*280+36))
    label=f"NSD {i}"+(f"; native pick {r['candidate']}/15" if col==2 else "")
    d.text((col*280+12,row*280+294),label,fill="black")
  buffer=io.BytesIO();im.save(buffer,format="JPEG",quality=95)
  path=atlas/f"fixed32-page-{pg+1}.jpg"
  if path.exists():assert path.read_bytes()==buffer.getvalue(),"Fixed atlas changed"
  else:path.write_bytes(buffer.getvalue())
 audit=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),n_eval=len(ids),formal_candidates=len(ids)*16,K=16,generation_exit=0,scoring_exit=0,all_candidate_sha_shape_range_finite_passed=True,all_selected_png_sha_passed=True,atlas_sha256={p.name:sha(p) for p in sorted(atlas.glob("*.jpg"))},review_status="Requires real AI review for initial100; full atlas must match prior reviewed bytes")
 (B/f"mindeye-native16-{a.pool}-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
 keys=["PixCorr","SSIM","Alex(2)","Alex(5)","Incep","CLIP","Eff","SwAV"]
 k1=json.loads((B/"metrics/mindeye1"/a.pool/"metrics.json").read_text())
 lines=["# MindEye1作者原生16候选：独立B表","",f"评价池：{len(ids)}个锁定标准测试目标；每图生成16候选，原作者脑预测CLIP相似度选择1张。选择器不使用GT。","",
 "| 协议 | PixCorr ↑ | SSIM ↑ | AN2 ↑ | AN5 ↑ | Incep ↑ | CLIP ↑ | Eff ↓ | SwAV ↓ |","|---|"+"---:|"*8]
 for label,obj in [("A表K1参考（已发布PNG评价）",k1),("B表原生16选1（原张量评价）",met)]:
  lines.append("| "+label+" | "+" | ".join(f'{obj["mean"][k]:.6f}' for k in keys)+" |")
 lines+=["","A行仅为参考，B结果未写入K1主表。完整高层prior和低层img2img保留；20步CFG3.5、strength0.85、prior100、每图seed42，原batch16，不按质量挑参数。原utils.reconstruction选择器只比较脑预测proj_embeddings与候选CLIP表示；GT仅用于评价、图册。","",
 "全部原16候选张量、作者选中下标和SHA均保留，逐文件SHA/形状/范围/有限值已核验，生成和评分exit0。B用原候选张量castFP32评价，PNG仅展示；A此前使用PNG评价。候选批量改变随机流且序列化精度不同，因此不是严格的纯K消融，也不声称逐配置复现论文数值。该100图与完整982负例池独立，不拼接评分。","",
 "固定32图由预先冻结列表决定；展示作者脑预测选择的候选，不由AI或GT重新选图。AI非盲观察另附。",""]
 for pg in range(1,5):lines.append(f"![固定32图第{pg}页](comparison_grid_mindeye_native16/fixed32-page-{pg}.jpg)")
 dest=B/f"MINDEYE_NATIVE16_{a.pool}_ZH.md";assert not dest.exists();dest.write_text("\n".join(lines)+"\n")
 print(json.dumps(audit))
if __name__=="__main__":main()
