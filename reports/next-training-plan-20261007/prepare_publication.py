from pathlib import Path
import re,html,json,hashlib,csv,xml.etree.ElementTree as ET
from urllib.parse import urljoin
out=Path(__file__).parent
base='https://github.com/GeYugong/neuroadapter-author-first/blob/author-first/reports/next-training-plan-20261007/'
doc=(out/'PLAN_ZH.md').read_text(encoding='utf-8')
def inline(s):
    s=html.escape(s,quote=False)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<a href="'+html.escape(urljoin(base,m[2]),quote=True)+'">'+m[1]+'</a>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    return s
lines=doc.splitlines();blocks=[];i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('# '):blocks.append('<h1>'+inline(line[2:])+'</h1>');i+=1;continue
    if line.startswith('## '):blocks.append('<h2>'+inline(line[3:])+'</h2>');i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
            i+=1
        assert all(len(row)==len(rows[0]) for row in rows)
        blocks.append('<table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])+'</tbody></table>');continue
    if line.startswith('- '):
        elems=[]
        while i<len(lines) and lines[i].startswith('- '):elems.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
        blocks.append('<ul>'+''.join(elems)+'</ul>');continue
    blocks.append('<p>'+inline(line)+'</p>');i+=1
xml='\n'.join(blocks)+'\n'
ET.fromstring('<root>'+xml+'</root>')
(out/'feishu-append.xml').write_bytes(xml.encode('utf-8'))
evidence=json.loads((out/'evidence.json').read_text(encoding='utf-8-sig'))
assert evidence['git_head']=='8f2b17c6bc26e03ba023dd9bf9a8edf1f998a4b5'
assert evidence['tracked_status']==''
assert all(x['complete'] for x in evidence['existing_checkpoint_inventory'].values())
assert abs(evidence['mean_seconds_epochs_301_to_500']-878.218)<.001
assert 400*532==212800 and 450*532==239400 and 500*532==266000
assert 50*532==26600 and 100*532==53200
# Validate the displayed validation table against existing raw CSV, without rescoring models.
repo=next((p for p in out.parents if (p/'train_brain_adapter.py').is_file()),out.parent/'neuroadapter-author-first')
records={r['epoch']:r for r in csv.DictReader((repo/'reports/training-diagnostics-20261006/validation_metrics.csv').open(encoding='utf-8-sig'))}
keys={'S':'S','PixCorr':'PixCorr','SSIM':'SSIM','AN2':'Alex(2)','AN5':'Alex(5)','Incep':'Incep','CLIP':'CLIP','Eff':'Eff','SwAV':'SwAV'}
for line in lines:
    if not line.startswith('|'):continue
    cols=[c.strip() for c in line.strip('|').split('|')]
    key=cols[0].split(' ')[0]
    if key not in keys or len(cols)!=5:continue
    assert cols[1:]==[format(float(records[epoch][keys[key]]),'.5f') for epoch in ['300','400','450','500']],key
# Normalize only our new files to UTF-8 LF.
for name in ['PLAN_ZH.md','evidence.json','collect_evidence.py','prepare_publication.py']:
    p=out/name;s=p.read_text(encoding='utf-8-sig');p.write_bytes(('\n'.join(x.rstrip() for x in s.splitlines())+'\n').encode('utf-8'))
print(json.dumps({'xml_valid':True,'top_level_blocks':len(blocks),'existing_metric_table_verified':True,'plan_sha256':hashlib.sha256((out/'PLAN_ZH.md').read_bytes()).hexdigest(),'new_training_updates':0},ensure_ascii=False))
