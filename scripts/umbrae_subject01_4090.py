"""Official UMBRAE S1 caption inference and pinned BrainHub evaluation, without training."""
import argparse,ast,copy,fcntl,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
O=R/'reports/umbrae-subject01-20261021-4090'
_MODELS_CACHE=None
_MEMFD_FILES=[]
A=R/'runtime/umbrae/assets'
U=R/'runtime/umbrae/upstream/UMBRAE/umbrae'
B=R/'runtime/umbrae/upstream/BrainHub'
PY=R/'runtime/env/bin/python'
PROMPT='Describe this image <image> as simply as possible.'
SYSTEM='A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER:'
GEN=dict(use_cache=True,do_sample=False,pad_token_id=2,bos_token_id=1,eos_token_id=2,max_new_tokens=512)
os.environ.update(HF_HOME=str(R/'runtime/umbrae/cache/hf'),TORCH_HOME=str(R/'runtime/umbrae/cache/torch'),
    XDG_CACHE_HOME=str(R/'runtime/umbrae/cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    HF_HUB_DISABLE_IMPLICIT_TOKEN='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='42')
sys.path.insert(0,str(U))

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value,fresh=False):
    p=O/name;assert p.resolve().is_relative_to(O.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    if fresh:
        with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    else:
        t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def loadjson(name):return json.loads((O/name).read_text())
def disk():
    free=shutil.disk_usage(R).free/2**30
    assert free>25,free
    return free
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def check_assets():
    plan=loadjson('asset-plan.json');checks={}
    for row in plan['files']:
        if 'model-000' in row['path']:continue
        p=R/row['path'];assert p.is_file() and p.stat().st_size==row['bytes'],row['path']
        assert sha(p)==row['sha256'],row['path'];checks[row['path']]=row['sha256']
    extras=json.loads((A/'evaluation-extras/manifest.json').read_text())
    for row in extras:
        p=A/'evaluation-extras'/row['name'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        checks[str(p.relative_to(R))]=row['sha256']
    return checks

def prepare():
    import numpy as np,torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor,to_pil_image
    torch.set_num_threads(4)
    assert not (O/'prepared.json').exists()
    checks=check_assets();disk()
    frozen_ids=[int(x) for x in (R/'benchmarks/subj01-v1/test_full_ids.txt').read_text().split()]
    assert len(frozen_ids)==len(set(frozen_ids))==982
    all_images=torch.load(A/'official/all_images.pt',map_location='cpu',weights_only=True)
    assert len(all_images)==982
    refs=json.loads((B/'data/caption/fmri_cococap.json').read_text())
    assert set(refs)=={str(i) for i in range(982)}
    groups={};order=[];source=[]
    for shard in range(2):
        p=A/'nsd-test'/f'test_subj01_{shard}.tar'
        with tarfile.open(p,'r') as tar:
            for member in tar:
                name=member.name
                suffix=next((s for s in ['.nsdgeneral.npy','.coco73k.npy','.jpg','.png'] if name.endswith(s)),None)
                if suffix is None:continue
                key=name[:-len(suffix)];identity=(shard,key)
                if identity not in groups:groups[identity]={};order.append(identity)
                assert suffix not in groups[identity]
                data=tar.extractfile(member).read()
                groups[identity][suffix]=data
    assert len(order)==982
    brains=[];rows=[];image_dir=O/'evaluation-images';image_dir.mkdir(exist_ok=False)
    for index,key in enumerate(order):
        g=groups[key];brain=np.load(io.BytesIO(g['.nsdgeneral.npy']),allow_pickle=False)
        nsd_id=int(np.load(io.BytesIO(g['.coco73k.npy']),allow_pickle=False).reshape(-1)[0])
        assert brain.shape==(3,15724) and brain.dtype==np.float16 and np.isfinite(brain).all()
        image_bytes=g.get('.jpg',g.get('.png'))
        image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
        native=to_tensor(image)
        reference=all_images[index].squeeze(0)
        assert native.shape==reference.shape and torch.equal(native,reference),f'BrainHub image order mismatch at {index}'
        assert len(refs[str(index)])>=1 and all(isinstance(s,str) and s.strip() for s in refs[str(index)])
        dest=image_dir/f'{index}.png';to_pil_image(reference).save(dest)
        brains.append(brain)
        rows.append(dict(native_index=index,nsd_id0=nsd_id,shard=key[0],sample=key[1],
            brain_sha256=hashlib.sha256(brain.tobytes()).hexdigest(),jpeg_sha256=hashlib.sha256(image_bytes).hexdigest(),
            brainhub_tensor_sha256=tensor_sha(reference),evaluation_png_sha256=sha(dest)))
    assert len({x['nsd_id0'] for x in rows})==982 and {x['nsd_id0'] for x in rows}==set(frozen_ids)
    brain_tensor=torch.from_numpy(np.stack(brains))
    means=brain_tensor.mean(1).float()
    assert means.shape==(982,15724) and torch.isfinite(means).all()
    with (O/'brain-inputs.npz').open('xb') as f:np.savez(f,brain=brain_tensor.numpy(),mean=means.numpy(),ids=np.array([x['nsd_id0'] for x in rows]))
    write('input-index.json',rows,fresh=True)
    # Independently compare the bounded reader with the author's WebDataset decoding.
    import webdataset as wds
    urls=[str(A/'nsd-test'/f'test_subj01_{s}.tar') for s in range(2)]
    dataset=wds.WebDataset(urls,resampled=False,nodesplitter=lambda urls:urls).decode('torch').rename(
        voxels='nsdgeneral.npy',images='jpg;png',coco='coco73k.npy').to_tuple('voxels','images','coco')
    count=0
    for count,(v,im,coco) in enumerate(dataset,1):
        ix=count-1;v=torch.as_tensor(v)
        assert torch.equal(v,brain_tensor[ix]) and int(torch.as_tensor(coco).reshape(-1)[0])==rows[ix]['nsd_id0']
        assert torch.equal(torch.mean(v.unsqueeze(0),axis=1).float()[0],means[ix])
    assert count==982
    checkpoint=torch.load(A/'official/train_logs/brainx-v-1-4/last.pth',map_location='cpu',weights_only=False)
    state=checkpoint['model_state_dict']
    from model import BrainX
    config=dict(hidden_dim=1024,out_dim=1024,num_latents=256,use_token=False,use_norm=False)
    model=BrainX(**config)
    result=model.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point())
    proj=torch.load(A/'official/model_weights/mm_projector.bin',map_location='cpu',weights_only=True)
    assert set(proj)=={'model.mm_projector.weight','model.mm_projector.bias'}
    code=[Path(__file__),R/"scripts/run_umbrae_official_score.py",U/'model.py',U/'perceiver.py',U/'utils.py',U/'inference.py',
        B/'eval_caption.py',*[B/'metrics'/n for n in ['__init__.py','clip_score.py','pac_score.py','sentence_score.py']]]
    plan=dict(subject=1,n=982,model='Official UMBRAE brainx-v-1-4, cross-subject model evaluated on S1',
        brain_config=config,prompt=PROMPT,seed=42,generation=GEN,
        precision='Brain FP32 parameters with FP16 autocast; projector FP32; Shikra FP16 parameters with FP16 autocast (author-recommended memory mode)',
        sharding='Native test order on GPU0; greedy decoding, batch1',
        streamed_assets=[r for r in loadjson('asset-plan.json')['files'] if 'model-000' in r['path']],
        streamed_source_host='5090; immutable original files remain under its project root',
        no_training=True,no_test_selection=True,no_GT_input=True,DREAM_untouched=True,
        source_hashes={str(p.relative_to(R)):sha(p) for p in code},asset_hashes=checks,
        input_hashes={n:sha(O/n) for n in ['brain-inputs.npz','input-index.json','asset-plan.json']},
        reference_captions_sha256=sha(B/'data/caption/fmri_cococap.json'),
        metrics=['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','ROUGE-L','CIDEr','SPICE','CLIPScore','RefCLIPScore','PACScore','RefPACScore','SentenceScore'],
        aggregation='Original corpus scorers for BLEU/METEOR/ROUGE/CIDEr/SPICE; original global means/F-scores for neural evaluators',
        limitations=['Official cross-subject checkpoint, not a newly trained Subject1-only model',
            'Modern Torch/CUDA and FP16 Shikra memory mode differ from historical full-precision loading',
            'Caption evaluation uses BrainHub author image tensors and references; these are not the eight image reconstruction metrics'])
    write('plan.json',plan,fresh=True)
    write('prepared.json',dict(complete=True,n=982,brain_shape=list(brain_tensor.shape),
        official_image_tensor_order_exact=True,original_WebDataset_and_mean_exact=True,
        locked_S1_test_ID_set_exact=True,reference_caption_count=sum(map(len,refs.values())),
        brain_state_strict_load=True,Shikra_load_pending=True,checkpoint_epoch=int(checkpoint['epoch']),
        free_gib=disk(),time=time.time()),fresh=True)
    print('PREPARATION_COMPLETE',flush=True)

def frozen(full=False):
    p=loadjson('plan.json')
    for field in ['source_hashes','input_hashes']:
        root=R if field=='source_hashes' else O
        for name,digest in p[field].items():assert sha(root/name)==digest,name
    if full:
        for name,digest in p['asset_hashes'].items():assert sha(R/name)==digest,name
    disk();return p

def models():
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:return _MODELS_CACHE
    import torch,numpy as np,random
    from model import BrainX
    from transformers import LlamaTokenizer,LlamaForCausalLM
    import utils
    torch.set_num_threads(4);utils.seed_everything(42,cudnn_deterministic=False)
    plan=frozen();materialize_shards(plan);wait_for_gpu0();brain=BrainX(**plan['brain_config'])
    ckpt=torch.load(A/'official/train_logs/brainx-v-1-4/last.pth',map_location='cpu',weights_only=False)
    brain.load_state_dict(ckpt['model_state_dict'],strict=True);brain.eval().requires_grad_(False).cuda();del ckpt
    tokenizer=LlamaTokenizer.from_pretrained(str(A/'official/model_weights/shikra-7b'),padding_side='left',local_files_only=True)
    model,info=LlamaForCausalLM.from_pretrained(str(A/'official/model_weights/shikra-7b'),torch_dtype=torch.float16,
        local_files_only=True,output_loading_info=True,attn_implementation='eager',low_cpu_mem_usage=True)
    assert not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs'],info
    assert set(info['unexpected_keys'])=={'model.mm_projector.weight','model.mm_projector.bias'},info
    model.eval().requires_grad_(False).cuda()
    release_shards()
    proj=torch.nn.Linear(1024,4096)
    state=torch.load(A/'official/model_weights/mm_projector.bin',map_location='cpu',weights_only=True)
    proj.load_state_dict({k.split('.')[-1]:v for k,v in state.items()},strict=True);proj.eval().requires_grad_(False).cuda()
    _MODELS_CACHE=(brain,proj,tokenizer,model,info)
    return _MODELS_CACHE

def features(brain,proj,raw):
    import torch
    voxel=torch.mean(raw,axis=1).float().cuda()
    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):encoded=brain(voxel,modal='fmri1')
    with torch.no_grad():projected=proj(encoded.float())
    assert projected.shape==(1,256,4096) and torch.isfinite(projected).all()
    return projected

def caption(model,tokenizer,feature):
    import torch,utils
    text=SYSTEM+PROMPT.replace('<image>',' <im_start>'+'<im_patch>'*256+'<im_end> ')+' ASSISTANT:'
    ids=tokenizer(text,return_tensors='pt').input_ids.cuda()
    base=model.model.embed_tokens(ids)
    starts=torch.where(ids[0]==32001)[0];assert len(starts)==1
    pos=int(starts[0]);assert ids[0,pos+257].item()==32002
    embeds=torch.cat((base[0,:pos+1],feature[0],base[0,pos+257:]),dim=0).unsqueeze(0)
    with torch.inference_mode(),torch.autocast(dtype=torch.float16,device_type='cuda'):
        output=model.generate(inputs_embeds=embeds.float(),**GEN)
    raw=tokenizer.batch_decode(output)[0]
    _,clean=utils.extract_id_bbox_caption(raw)
    assert isinstance(clean,str) and clean.strip()
    return dict(raw_response=raw,caption=clean,token_ids=output[0].cpu().tolist(),feature_sha256=tensor_sha(feature))

def preflight(rank):
    import numpy as np,torch,utils
    brain,proj,tokenizer,model,info=models()
    with np.load(O/'brain-inputs.npz') as z:raw=torch.from_numpy(z['brain'][:2].copy())
    ff=torch.cat([features(brain,proj,raw[i:i+1]) for i in range(2)])
    ours=[caption(model,tokenizer,ff[i:i+1]) for i in range(2)]
    # Execute the upstream caption loop, with only token recording inserted.
    tree=ast.parse((U/'inference.py').read_text())
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cap_result' for t in n.targets))
    body=ast.Module(body=tree.body[start:],type_ignores=[])
    count=0
    class Capture(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal count
            if any(isinstance(t,ast.Name) and t.id=='response' for t in node.targets):
                count+=1
                return [node,ast.parse('native_tokens[cur_image_idx] = output_ids[0].cpu().tolist()').body[0]]
            return node
    body=ast.fix_missing_locations(Capture().visit(body));assert count==1
    text=SYSTEM+PROMPT.replace('<image>',' <im_start>'+'<im_patch>'*256+'<im_end> ')+' ASSISTANT:'
    ids=tokenizer(text,return_tensors='pt').input_ids.cuda()
    target=O/f'preflight-native-{rank}';target.mkdir(exist_ok=False)
    ctx=dict(torch=torch,os=os,json=json,time=time,utils=utils,model=model,tokenizer=tokenizer,
        image_features=ff,input_ids=ids,inputs_embeds=model.model.embed_tokens(ids),gen_kwargs=GEN,
        result_dir=str(target),save_image=False,native_tokens={})
    exec(compile(body,str(U/'inference.py'),'exec'),ctx)
    for i in range(2):
        assert ctx['native_tokens'][i]==ours[i]['token_ids']
        assert ctx['cap_result'][str(i)]==ours[i]['caption']
    write(f'preflight-{rank}.json',dict(passed=True,rank=rank,original_loop_tokens_exact=True,
        samples=ours,loading_info=info,peak_cuda_bytes=torch.cuda.max_memory_allocated(),time=time.time()),fresh=True)
    print('PREFLIGHT_PASSED',rank,flush=True)

def generate(rank):
    import numpy as np,torch
    brain,proj,tokenizer,model,info=models();rows=loadjson('input-index.json')
    with np.load(O/'brain-inputs.npz') as z:raw=torch.from_numpy(z['brain'].copy())
    (O/'records').mkdir(exist_ok=True)
    for index in range(982):
        path=O/'records'/f'{index:04}.json';assert not path.exists(),'No automatic regeneration'
        feature=features(brain,proj,raw[index:index+1]);result=caption(model,tokenizer,feature)
        result.update(rows[index]);result.update(rank=rank,seed=42,prompt=PROMPT,GT_used=False,time=time.time())
        write(f'records/{index:04}.json',result,fresh=True)
        write(f'progress-{rank}.json',dict(completed=index+1,total=982,last_native_index=index,time=time.time()))
        print('CAPTION',rank,index,result['caption'],flush=True)
    write(f'generation-summary-{rank}.json',dict(complete=True,n=982,peak_cuda_bytes=torch.cuda.max_memory_allocated(),loading_info=info),fresh=True)

def evaluate():
    from run_umbrae_official_score import score
    result=score()
    frozen(full=True)
    write('final-audit.json',dict(complete=True,n=982,metrics=13,
        score_source='Official BrainHub main entry',metrics_file='final-metrics.json',
        assets_and_sources_unchanged=True,time=time.time()),fresh=True)



def wait_for_gpu0():
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        first=next(x for x in rows.splitlines() if x.split(',')[0].strip()=='0')
        _,uid,free=[x.strip() for x in first.split(',')]
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        occupied=[x for x in jobs.splitlines() if x.startswith(uid+',') and int(x.split(',')[1])!=os.getpid()]
        if not occupied and int(free)>=22000:return
        write('status.json',dict(phase='waiting_for_gpu0',free_mib=int(free),time=time.time()))
        time.sleep(30)

def materialize_shards(plan):
    import gc,torch
    from safetensors import safe_open
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>40*2**30,'At least40GiB available RAM required for memory-backed loading'
    fifo=O/'model-weights.fifo';assert not fifo.exists();os.mkfifo(fifo,0o600)
    records=[]
    for row in plan['streamed_assets']:
        target=R/row['path'];assert not target.exists() and not target.is_symlink()
        fd=os.memfd_create(target.name,flags=0);h=hashlib.sha256();received=0
        write('stream-status.json',dict(phase='waiting_for_shard',path=row['path'],bytes=row['bytes'],time=time.time()))
        with fifo.open('rb',buffering=0) as source,os.fdopen(os.dup(fd),'wb') as sink:
            while chunk:=source.read(4*2**20):
                received+=len(chunk);assert received<=row['bytes'];sink.write(chunk);h.update(chunk)
        assert received==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
        target.symlink_to(f'/proc/{os.getpid()}/fd/{fd}');_MEMFD_FILES.append((target,fd))
        records.append(dict(path=row['path'],bytes=received,sha256=h.hexdigest(),verified=True,storage='anonymous_RAM'))
        write('stream-status.json',dict(phase='received',path=row['path'],time=time.time()))
    # Both projector copies must agree before the checkpoint's extra keys are ignored.
    shikra=A/'official/model_weights/shikra-7b'
    mapping=json.loads((shikra/'model.safetensors.index.json').read_text())['weight_map']
    proj=torch.load(A/'official/model_weights/mm_projector.bin',map_location='cpu',weights_only=True)
    for key,value in proj.items():
        with safe_open(str(shikra/mapping[key]),framework='pt',device='cpu') as f:
            assert torch.equal(f.get_tensor(key).float(),value.float())
    write('stream-audit.json',dict(all_three_verified=True,projector_matches_Shikra=True,shards=records),fresh=True)
    fifo.unlink()

def release_shards():
    for path,fd in _MEMFD_FILES:
        assert path.is_symlink() and str(path.readlink())==f'/proc/{os.getpid()}/fd/{fd}'
        path.unlink();os.close(fd)
    _MEMFD_FILES.clear()

def run():
    global _MODELS_CACHE
    import torch,gc
    with (O/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (O/'started.json').exists(),'Inspect existing attempt before any restart'
        write('started.json',dict(pid=os.getpid(),gpu=0,time=time.time()),fresh=True)
        write('status.json',dict(phase='loading_weights_into_RAM',time=time.time()))
        preflight(0)
        write('status.json',dict(phase='generating_captions',gpu=0,n=982,time=time.time()))
        generate(0)
        _MODELS_CACHE=None;gc.collect();torch.cuda.empty_cache()
        write('status.json',dict(phase='scoring_13_metrics',time=time.time()))
        evaluate()
        write('status.json',dict(phase='complete',n=982,metrics=13,time=time.time()))
        write('exit.json',dict(returncode=0,time=time.time()),fresh=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','generate','evaluate','run']);p.add_argument('--rank',type=int,default=0);a=p.parse_args()
    try:
        {'prepare':prepare,'preflight':lambda:preflight(a.rank),'generate':lambda:generate(a.rank),'evaluate':evaluate,'run':run}[a.mode]()
    except BaseException:
        write(f'failure-{a.mode}-{a.rank}-{time.time_ns()}.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
