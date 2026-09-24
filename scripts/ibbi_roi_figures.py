"""Render every fixed trajectory/condition; display-only use of the author's overlay."""
import ast,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw,ImageFont
from ibbi_roi_prepare import ROOT,RUN,REPORT,WORK
import lr_half_evaluate as old
import evaluate_stage as ev
GROUP_NAMES=['V1','V2','V3','V4','Face','Body','Scene','Word']
def author_overlay():
    tree=ast.parse((ROOT/'plot_attn_dynamics.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_attention_overlay']
    ns={'np':np,'plt':plt,'ListedColormap':ListedColormap,'gaussian_filter':gaussian_filter}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'author_overlay','exec'),ns)
    return ns['create_attention_overlay']
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
def text(draw,xy,s):draw.text(xy,s,fill='black',font=FONT)
def to_gt(p):return (p['groundtruth_image'].transpose(1,2,0)*255).round().astype('uint8')
def tile(canvas,img,x,y,size=160):
    canvas.paste(Image.fromarray(np.asarray(img,dtype=np.uint8)).resize((size,size)),(x,y))
def attention_figures():
    overlay=author_overlay();out=REPORT/'attention';out.mkdir(exist_ok=True)
    ids=old.ids('atlas_val32_ids.txt');records=[];all_b=[];means={}
    for i in ids:
        with np.load(WORK/f'ibbi32/image_{i:06d}.npz') as p:
            gt=to_gt(p);candidates=p['candidate_images'].copy()
        for k in (0,1):
            trace=WORK/f'ibbi32/attention/image_{i:06d}-candidate-{k}.npz'
            with np.load(trace) as a:
                b=a['brain_attention'];layer_b=a['layer_brain_attention'];ram=a['roi_spatial_maps']
                steps=a['saved_step_indices'];times=a['actual_timesteps']
                assert b.shape==(50,200) and ram.shape==(7,8,512,512)
                assert np.allclose(b.sum(axis=1),1,atol=2e-5)
                lite=out/f'image_{i:06d}-candidate-{k}-quantitative.npz'
                np.savez_compressed(lite,brain_attention=b,layer_brain_attention=layer_b,
                    actual_timesteps=times,layer_names=a['layer_names'],group_names=a['group_names'],
                    roi_spatial_maps_128=ram.reshape(7,8,128,4,128,4).sum(axis=(3,5)))
                # Quantitative 128 maps use exact nonoverlapping 4x4 mass sums, never display minmax.
                all_b.append(b);means[i,k]=ram[-1].copy()
                fig,ax=plt.subplots(figsize=(12,4))
                im=ax.imshow(b,aspect='auto',origin='upper',cmap='viridis',vmin=0)
                ax.axvline(99.5,color='white',lw=.7);ax.set_xlabel('Selected parcel token: LH 0-99 | RH 100-199')
                ax.set_ylabel('Denoising step (1-50)');ax.set_yticks([0,9,19,29,39,49],['1','10','20','30','40','50'])
                ax.set_title(f'L120 | image {i} | candidate {k} | model-relative attention share')
                fig.colorbar(im,ax=ax,label='B_t (query-weighted layers)')
                fig.text(.02,.01,'Only saved 200 parcels assessed. All other cortical vertices: NOT EVALUATED.',fontsize=9)
                fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/f'image_{i:06d}-candidate-{k}-brain.png',dpi=110);plt.close(fig)
                canvas=Image.new('RGB',(1600,55+7*188),'white');draw=ImageDraw.Draw(canvas)
                text(draw,(8,5),f'L120 image {i} candidate {k}: author display overlay; raw probability arrays saved separately')
                for c,title in enumerate(['GT','Generated']+GROUP_NAMES):text(draw,(c*160+5,30),title)
                for r,step in enumerate(steps):
                    y=55+r*188;tile(canvas,gt,0,y);tile(canvas,candidates[k],160,y)
                    for g in range(8):
                        if np.isfinite(ram[r,g]).all():img=overlay(candidates[k],ram[r,g])
                        else:img=np.full_like(candidates[k],220)
                        tile(canvas,img,(g+2)*160,y)
                    label='All50 normalized-step mean' if step==-1 else f'Step {int(step)} / scheduler t={int(times[step-1])}'
                    text(draw,(6,y+162),label)
                page=out/f'image_{i:06d}-candidate-{k}-ram.jpg';canvas.save(page,quality=88)
                records.append({'image_id':i,'candidate':k,'trace_server_path':str(trace.resolve()),
                    'trace_sha256':old.sha(trace),'quantitative_summary':lite.name,'ram_display':page.name,
                    'brain_display':f'image_{i:06d}-candidate-{k}-brain.png',
                    'trace_check':old.read(trace.with_suffix('.json'))})
    all_b=np.stack(all_b);np.savez_compressed(out/'all-brain-attention.npz',image_ids=np.repeat(ids,2),
        candidate_ids=np.tile([0,1],len(ids)),brain_attention=all_b)
    # Eight pages contain every generated candidate and every ROI mean, in the fixed order.
    for part in range(8):
        chosen=ids[part*4:(part+1)*4];canvas=Image.new('RGB',(1600,50+len(chosen)*2*188),'white');draw=ImageDraw.Draw(canvas)
        for c,title in enumerate(['GT','Generated']+GROUP_NAMES):text(draw,(c*160+5,12),title)
        for r,i in enumerate(chosen):
            with np.load(WORK/f'ibbi32/image_{i:06d}.npz') as p:
                gt=to_gt(p);cand=p['candidate_images']
                for k in (0,1):
                    y=50+(2*r+k)*188;tile(canvas,gt,0,y);tile(canvas,cand[k],160,y)
                    for g in range(8):tile(canvas,overlay(cand[k],means[i,k][g]),(g+2)*160,y)
                    text(draw,(6,y+162),f'image_id={i}, candidate={k}; average of all50 per-step RAMs')
        canvas.save(out/f'all-candidates-{part+1}.jpg',quality=88)
    ev.atlas(WORK/'ibbi32',REPORT/'ibbi32_ids.txt',REPORT/'ibbi-reconstruction-atlas')
    old.save(out/'manifest.json',{'trajectories':records,'full_spatial_arrays':'Each server trace has unscaled float32 512x512 maps for steps1/10/20/30/40/50 and mean.',
        'published_spatial_summary':'128x128 exact 4x4 block probability sums; normalized mass preserved. Not used for scoring.',
        'display':'Original author create_attention_overlay: Gaussian sigma10 + logistic contrast + viridis alpha; not a quantitative minmax array.',
        'unselected_cortex':'Not evaluated; heatmaps cover only saved selected200 parcel tokens.'})
    lines=['# All fixed IBBI trajectories','','Display overlays are model-internal relative attention, not brain causality. Quantitative files do not contain display contrast scaling.','']
    for part in range(8):lines+=['![](all-candidates-'+str(part+1)+'.jpg)','']
    for i in ids:
        lines+=['## Image '+str(i),'']
        for k in (0,1):lines+=[f'Candidate {k}: [unscaled arrays](image_{i:06d}-candidate-{k}-quantitative.npz)',f'![](image_{i:06d}-candidate-{k}-brain.png)',f'![](image_{i:06d}-candidate-{k}-ram.jpg)','']
    (out/'README.md').write_text('\n'.join(lines))
def intervention_figures():
    names=['BASE','LL','HL','LL-R1','LL-R2','LL-R3','HL-R1','HL-R2','HL-R3']
    available=old.read(REPORT/'roi_mapping.json')['mask_sets'];names=[n for n in names if n=='BASE' or n in available]
    out=REPORT/'all-condition-atlas';out.mkdir(exist_ok=True);ids=old.ids('atlas_val32_ids.txt')
    for k in (0,1):
        for part in range(4):
            selected=ids[part*8:(part+1)*8];canvas=Image.new('RGB',((len(names)+1)*160,50+len(selected)*184),'white');draw=ImageDraw.Draw(canvas)
            for c,title in enumerate(['GT']+names):text(draw,(c*160+5,12),title)
            for r,i in enumerate(selected):
                y=50+r*184
                with np.load(RUN/f'decoded/epoch-120/val/image_{i:06d}.npz') as b:tile(canvas,to_gt(b),0,y)
                for c,name in enumerate(names):
                    folder=RUN/'decoded/epoch-120/val' if name=='BASE' else WORK/name
                    with np.load(folder/f'image_{i:06d}.npz') as a:tile(canvas,a['candidate_images'][k],(c+1)*160,y)
                text(draw,(5,y+162),f'image_id={i} candidate={k}; fixed order, no selection')
            canvas.save(out/f'candidate-{k}-page-{part+1}.jpg',quality=92)
    (out/'README.md').write_text('# All fixed intervention reconstructions\n\n'+
        '\n\n'.join(f'![](candidate-{k}-page-{p}.jpg)' for k in (0,1) for p in range(1,5)))

