"""Freeze L120, official NSD ROI mapping and the bounded intervention masks."""
from pathlib import Path
import json,hashlib,urllib.request,csv,subprocess,time
import numpy as np
import nibabel as nib
import torch
import lr_half_evaluate as old
ROOT=old.ROOT; RUN=old.L
REPORT=ROOT/'reports/ibbi-roi-l120-pilot'
WORK=RUN/'comparison/ibbi-roi-l120-pilot'
ASSETS=ROOT/'runtime/ibbi-roi-l120-assets'
GROUPS={'V1':['V1v','V1d'],'V2':['V2v','V2d'],'V3':['V3v','V3d'],'V4':['hV4'],
'Face':['OFA','FFA-1','FFA-2','mTL-faces','aTL-faces'],
'Body':['EBA','FBA-1','FBA-2','mTL-bodies'],'Scene':['OPA','PPA','RSC'],
'Word':['OWFA','VWFA-1','VWFA-2','mfs-words','mTL-words']}
CATEGORIES=['prf-visualrois','floc-faces','floc-bodies','floc-places','floc-words']
def fetch(url):
    for attempt in range(3):
        try:return urllib.request.urlopen(url,timeout=45).read()
        except Exception:
            if attempt==2:raise
            time.sleep(2)
def prepare():
    assert not (REPORT/'model_lock.json').exists()
    REPORT.mkdir(exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    meta=old.read(RUN/'exports/epoch-120/metadata.json')
    vertices=torch.load(RUN/'exports/epoch-120/parcel_vertices.pt',weights_only=True,map_location='cpu')
    ident=old.identity(RUN,120,63840,RUN/'decoded/epoch-120/val')
    assert ident['sha256']=='bed56671f4e48ccf031e41c1daa36d473a4e325e94e3264aa1b21e08080fd798'
    source_commit='bfd36a503cc90a3eb3ebfd69b269403f7e186924'
    # Archive and inspect official conversion reference; do not install or change shared environments.
    reference={}
    for name in ['parse_case.py','transform_data.py','load_data.py']:
        url=f'https://raw.githubusercontent.com/cvnlab/nsdcode/{source_commit}/nsdcode/{name}'
        data=(ASSETS/name).read_bytes() if (ASSETS/name).exists() else fetch(url)
        (ASSETS/name).write_bytes(data);reference[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
    assert 'sourcedata[np.squeeze(a1_data.astype(int)) - 1]' in (ASSETS/'transform_data.py').read_text()
    nc=np.load(ROOT/'data/metadata_sub-01.npy',allow_pickle=True).item()
    labels={};mapped={};space={}
    for cat in CATEGORIES:
        labels[cat]={int(l.split()[0]):l.split()[1] for l in (ASSETS/f'{cat}.mgz.ctab').read_text().splitlines() if l.strip()}
        assert labels[cat][0]=='Unknown'
    for hemi in ['lh','rh']:
        idx=nib.load(ASSETS/f'{hemi}.white-to-fsaverage.mgz').get_fdata().squeeze()
        assert idx.shape==(163842,) and np.isfinite(idx).all() and np.array_equal(idx,np.round(idx))
        idx=idx.astype(np.int64)-1
        native_count=227021 if hemi=='lh' else 226601
        assert idx.min()>=0 and idx.max()<native_count
        source_parcels=torch.load(Path(meta['config']['parcel_dir'])/f'{hemi}_labels_s01.pt',weights_only=True)
        for j,vs in enumerate(vertices[hemi]):
            pi=meta[hemi]['parcel_indices'][j]
            assert torch.equal(vs,source_parcels[pi+1])
            assert len(vs)==meta[hemi]['voxel_counts'][j] and len(vs.unique())==len(vs)
            assert 0<=int(vs.min())<=int(vs.max())<163842
        # Verify the SNR metadata uses the same hemisphere-wise fsaverage vertex indexing.
        snr_path=ROOT.parent/'neuroadapter-subject1-research/data/raw/nsd/betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR'/f'{hemi}.ncsnr.mgh'
        raw=nib.load(snr_path).get_fdata().squeeze()
        assert raw.shape==(163842,) and np.allclose(raw,nc[f'{hemi}_ncsnr'].squeeze(),equal_nan=True)
        for cat in CATEGORIES:
            source=nib.load(ASSETS/f'{hemi}.{cat}.mgz').get_fdata().squeeze()
            assert source.shape==(native_count,) and np.isfinite(source).all()
            assert set(np.unique(source)).issubset(labels[cat])
            # NSD official transform_data case 3: nearest neighbor, 1-based stored index -> 0-based.
            mapped[hemi,cat]=source[idx].astype(np.int16)
        space[hemi]={'native_vertices':native_count,'target_vertices':len(idx),'source_index_range_zero_based':[int(idx.min()),int(idx.max())],'method':'official NSD surface-to-fsaverage nearest-neighbor indices; subtract 1','snr_fsaverage_exact_index_validation':True}
    allnames=[name for cat in CATEGORIES for v,name in labels[cat].items() if v]
    rows=[];groups={g:[] for g in GROUPS}
    for hi,hemi in enumerate(['lh','rh']):
        for j,vs in enumerate(vertices[hemi]):
            vi=vs.numpy();row={'token_index':hi*100+j,'hemisphere':hemi,'parcel_index_after_medial_wall_removed':meta[hemi]['parcel_indices'][j],
             'original_schaefer_label_id':meta[hemi]['parcel_indices'][j]+1,'valid_vertex_count':len(vi),
             'snr_mean':float(nc[f'{hemi}_ncsnr'].squeeze()[vi].mean()),'snr_median':float(np.median(nc[f'{hemi}_ncsnr'].squeeze()[vi]))}
            hits=[]
            for cat in CATEGORIES:
                for val,name in labels[cat].items():
                    if val:
                        ratio=float(np.mean(mapped[hemi,cat][vi]==val));row['overlap_'+name]=ratio
                        if ratio>.5:hits.append(name)
            assigned=[g for g,names in GROUPS.items() if set(hits)&set(names)]
            row.update(matched_raw_labels=';'.join(hits),groups=';'.join(assigned),LL=any(g in assigned for g in ['V1','V2','V3','V4']),HL=any(g in assigned for g in ['Face','Body','Scene','Word']))
            for g in assigned:groups[g].append(hi*100+j)
            rows.append(row)
    for g in ['LL','HL']:groups[g]=[x['token_index'] for x in rows if x[g]]
    with (REPORT/'parcel_roi_mapping.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rng=np.random.default_rng(2026092403);masks={k:groups[k] for k in ['LL','HL']};unavailable={}
    for g in ['LL','HL']:
        assert groups[g],g
        candidates={h:[i for i in range(h*100,(h+1)*100) if i not in groups[g]] for h in [0,1]}
        counts={h:sum(h*100<=i<(h+1)*100 for i in groups[g]) for h in [0,1]}
        if any(len(candidates[h])<counts[h] for h in [0,1]):unavailable[g]='Complement insufficient for hemisphere matching';continue
        seen=set()
        for k in [1,2,3]:
            while True:
                choice=tuple(sorted(sum([rng.choice(candidates[h],counts[h],replace=False).tolist() for h in [0,1]],[])))
                if choice not in seen:seen.add(choice);break
            masks[f'{g}-R{k}']=list(choice)
    summary={}
    for name,tokens in {**groups,**masks}.items():
        selected=[rows[t] for t in tokens]
        summary[name]={'n':len(tokens),'lh':sum(t<100 for t in tokens),'rh':sum(t>=100 for t in tokens),
          'valid_vertices_total':sum(x['valid_vertex_count'] for x in selected),
          'snr_mean_quantiles':np.quantile([x['snr_mean'] for x in selected],[0,.25,.5,.75,1]).tolist() if selected else [],
          'vertex_count_quantiles':np.quantile([x['valid_vertex_count'] for x in selected],[0,.25,.5,.75,1]).tolist() if selected else [],
          'functional_group_counts':{g:sum(t in groups[g] for t in tokens) for g in GROUPS}}
    old.save(REPORT/'roi_mapping.json',{'strict_overlap_rule':'>0.5 per raw label; no pre-union of labels; padding excluded','labels':labels,'group_definitions':GROUPS,
      'groups':groups,'mask_sets':masks,'random_mask_seed':2026092403,'group_summaries':summary,'unavailable_controls':unavailable,
      'multi_raw_label_tokens':[x['token_index'] for x in rows if ';' in x['matched_raw_labels']],
      'LL_HL_intersection':sorted(set(groups['LL'])&set(groups['HL'])),'conflict_policy':'Preserve every raw >0.5 membership; unions deduplicated within group; cross-group overlaps retained.',
      'space_validation':space,'nsdcode_reference':reference,'matching_limits':'Only parcel count and hemisphere counts exact; SNR, vertices, function and spatial adjacency not matched.'})
    lock={'subject':1,'branch':'L','epoch':120,'global_optimizer_updates':63840,'selection':'Existing val100 S rule; unchanged by ROI results','baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'files':{str(p.relative_to(ROOT)):old.sha(p) for p in [RUN/'exports/epoch-120'/n for n in ['inference.pt','metadata.json','parcel_vertices.pt']]},
      'baseline_identity':ident,'training_updates_added':0}
    old.save(REPORT/'model_lock.json',lock)
    protected=old.read(ROOT/'reports/brain-correspondence-l120/protected-before.json')
    for folder in [ROOT/'reports',ROOT/'configs']:
        for p in folder.rglob('*'):
            if p.is_file() and REPORT not in p.parents and str(p.relative_to(ROOT)) not in protected:protected[str(p.relative_to(ROOT))]=old.sha(p)
    for n in ['decode_brain_adapter.py','extract_brain_adapter.py','brain_adapter/ip_adapter/attention_processor.py','plot_attn_dynamics.py','brain_adapter/dataset.py','metric_brain_adapter.py','scripts/evaluate_stage.py']:
        protected[n]=old.sha(ROOT/n)
    old.save(REPORT/'protected-before.json',protected)
    for name,ids in [('ibbi32','atlas_val32_ids.txt'),('check2','decode2_ids.txt'),('pilot100','monitor100_ids.txt')]:
        (REPORT/f'{name}_ids.txt').write_text((ROOT/'configs'/ids).read_text())
    atlas=old.ids('atlas_val32_ids.txt');check=old.ids('decode2_ids.txt')
    assert set(check)<=set(atlas)
    (REPORT/'ibbi_remaining_ids.txt').write_text(''.join(str(i)+'\n' for i in atlas if i not in check))
    print(json.dumps({'groups':{g:len(t) for g,t in groups.items()},'controls':list(masks),'overlap':sorted(set(groups['LL'])&set(groups['HL'])),'protected_files':len(protected)}))
if __name__=='__main__':prepare()
