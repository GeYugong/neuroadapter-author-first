"""Descriptive unmatched SNR/vertex/function/adjacency characteristics of frozen masks."""
import urllib.request
import numpy as np
import nibabel as nib
import torch
from ibbi_roi_prepare import ROOT,RUN,REPORT,ASSETS
import lr_half_evaluate as old
def main():
    mapping=old.read(REPORT/'roi_mapping.json');before=old.sha(REPORT/'roi_mapping.json')
    v=torch.load(RUN/'exports/epoch-120/parcel_vertices.pt',weights_only=True,map_location='cpu')
    nc=np.load(ROOT/'data/metadata_sub-01.npy',allow_pickle=True).item()
    pairs=set();sources=[]
    for hi,h in enumerate(['lh','rh']):
        url=f'https://natural-scenes-dataset.s3.amazonaws.com/nsddata/freesurfer/fsaverage/surf/{h}.white'
        p=ASSETS/f'{h}.fsaverage.white'
        if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        coordinates,faces=nib.freesurfer.read_geometry(p)
        assert coordinates.shape==(163842,3) and faces.shape[1]==3
        labels=np.full(163842,-1,dtype=int)
        for j,vs in enumerate(v[h]):labels[vs.numpy()]=hi*100+j
        edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]])
        token_edges=labels[edges]
        valid=(token_edges[:,0]>=0)&(token_edges[:,1]>=0)&(token_edges[:,0]!=token_edges[:,1])
        pairs.update(map(tuple,np.unique(np.sort(token_edges[valid],axis=1),axis=0).tolist()))
        sources.append({'hemisphere':h,'url':url,'sha256':old.sha(p),'vertices':len(coordinates),
          'faces':len(faces),'adjacency':'Two selected parcels are adjacent if at least one white-surface mesh edge joins their vertices.'})
    result={}
    for name,tokens in {**mapping['groups'],**mapping['mask_sets']}.items():
        selected=set(tokens);vertices=[];snr=[]
        for i in tokens:
            h='lh' if i<100 else 'rh';vs=v[h][i%100].numpy()
            vertices.append(len(vs));snr.extend(nc[f'{h}_ncsnr'].reshape(-1)[vs].tolist())
        snr=np.array(snr);finite=snr[np.isfinite(snr)]
        within=[(a,b) for a,b in pairs if a in selected and b in selected]
        graph={t:set() for t in tokens}
        for a,b in within:graph[a].add(b);graph[b].add(a)
        unseen=set(tokens);components=[]
        while unseen:
            stack=[unseen.pop()];size=0
            while stack:
                x=stack.pop();size+=1
                for y in graph[x]&unseen:unseen.remove(y);stack.append(y)
            components.append(size)
        result[name]={'n_parcels':len(tokens),'n_vertices':sum(vertices),'finite_snr_vertices':len(finite),
          'nonfinite_snr_vertices':len(snr)-len(finite),'vertex_snr_quantiles':np.quantile(finite,[0,.25,.5,.75,1]).tolist() if len(finite) else [],
          'vertex_snr_mean':float(finite.mean()) if len(finite) else None,
          'selected_parcel_adjacent_pairs':len(within),'connected_component_sizes':sorted(components,reverse=True),
          'within_set_parcel_degrees':[len(graph[t]) for t in tokens],
          'functional_group_counts':mapping['group_summaries'][name]['functional_group_counts']}
    old.save(REPORT/'mapping-characteristics.json',{'groups_and_masks':result,'spatial_assets':sources,
        'matching':'Only parcel count and LH/RH counts exact. These SNR/size/function/adjacency measures are descriptive and were not used to choose masks.',
        'adjacency_scope':'Graph restricted to the model selected200 parcels; gaps through unselected cortex do not become inferred edges.'})
    assert old.sha(REPORT/'roi_mapping.json')==before
if __name__=='__main__':main()

