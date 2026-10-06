from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
data=json.loads((HERE/"matches.json").read_text())
paper=Image.open(HERE/"user-paper-examples.png").convert("RGB")
big=ImageFont.truetype(FONT,25);regular=ImageFont.truetype(FONT,22)
small=ImageFont.truetype(FONT,16);tiny=ImageFont.truetype(FONT,15)
manifest=[]
for page in range(4):
    pic=Image.new("RGB",(1050,2388),"white");d=ImageDraw.Draw(pic)
    d.text((18,10),f"论文同GT对照 · Subject1 · 第{page+1}/4页",font=big,fill="#172634")
    d.text((18,49),"按用户所附论文图顺序；直接读取既有完整ImageNet检索top1，未重跑或挑选候选。",font=small,fill="#344454")
    for col,label in enumerate(["GT（与论文同图）","论文baseline（附件截图）","我们的baseline（实测fMRI）"]):
        d.text((col*350+12,83),label,font=regular,fill="#172634")
    page_records=data["records"][page*6:(page+1)*6]
    for row,rec in enumerate(page_records):
        y=124+row*363
        d.text((14,y),f'论文例{rec["paper_position"]:02}  |  NSD ID0={rec["nsd_id0"]}  |  我们的Pearson={rec["pearson"]:.6f}  |  {rec["candidate_id"]}',font=small,fill="#172634")
        ims=[Image.open(rec["gt_path"]).convert("RGB"),paper.crop(rec["paper_baseline_box"]),Image.open(rec["retrieved_preview_path"]).convert("RGB")]
        for col,im in enumerate(ims):
            pic.paste(im.resize((332,332),Image.Resampling.LANCZOS),(col*350+9,y+26))
    d.text((14,2315),"本地固定协议；encoder开发集划分未核实；标准化及mask与作者历史实验的一致性未确认。",font=tiny,fill="#4d555b")
    d.text((14,2344),"论文栏仅为附件中的展示结果，未独立重算；GT原图与论文截图可能有轻微裁剪/缩放差异。",font=tiny,fill="#4d555b")
    p=HERE/f"page-{page+1:02}.jpg";pic.save(p,quality=95)
    manifest.append(dict(page=page+1,file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),positions=[z["paper_position"] for z in page_records],ids0=[z["nsd_id0"] for z in page_records]))
(HERE/"pages.json").write_text(json.dumps(manifest,indent=2)+"\n")
