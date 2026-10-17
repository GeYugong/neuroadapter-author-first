"""Publish the server allowlist through a bounded local spool.

Run on the local host with NEUROADAPTER_HF_RELEASE_DIR pointing to the release
report folder; NEUROADAPTER_HF_SPOOL optionally selects a roomier temporary disk.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(os.environ.get('NEUROADAPTER_HF_RELEASE_DIR', str(Path(__file__).resolve().parent))).resolve()
SPOOL = Path(os.environ.get('NEUROADAPTER_HF_SPOOL', str(HERE / 'staging'))).resolve()
SPOOL.mkdir(parents=True, exist_ok=True)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_XET_CACHE'] = str(HERE / 'xet-cache')
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
from huggingface_hub import HfApi, CommitOperationAdd
import requests

ROOT = '/data1/matengyu/geyugong/neuroadapter-author-first'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=15', 'neuroadapter-4090']
MANIFEST = HERE / 'upload-manifest.json'
EVENTS = HERE / 'upload-events.jsonl'


def event(**item):
    item['unix_time'] = time.time()
    line = json.dumps(item, ensure_ascii=False)
    with EVENTS.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
    print(line, flush=True)


def ssh_capture(command):
    p = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=90)
    if p.returncode:
        raise RuntimeError('SSH read failed')
    return p.stdout.strip()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def public_metadata(api, repo, path):
    values = api.get_paths_info(repo_id=repo, paths=[path], repo_type='model', token=False)
    return next((v for v in values if v.path == path), None)


def check_public(api, repo, item):
    value = public_metadata(api, repo, item['path'])
    if value is None or value.size != item['size']:
        return False
    lfs = getattr(value, 'lfs', None)
    if lfs:
        digest = lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256
        return digest == item['sha256']
    if item['size'] > 8 * 1024 * 1024:
        raise RuntimeError('Large public weight has no LFS hash')
    response = requests.get(f"https://huggingface.co/{repo}/resolve/main/{item['path']}", timeout=60)
    response.raise_for_status()
    return hashlib.sha256(response.content).hexdigest() == item['sha256']


def download_one(item):
    destination = SPOOL / (item['sha256'] + Path(item['path']).suffix)
    if destination.exists() and destination.stat().st_size == item['size'] and sha(destination) == item['sha256']:
        return destination
    available = shutil.disk_usage(SPOOL).free
    already = destination.stat().st_size if destination.exists() else 0
    if available + already < item['size'] + 3 * 1024**3:
        raise RuntimeError('Insufficient bounded spool space')
    # Paths are from our explicit fixed manifest and cannot contain shell syntax.
    source = item['source']
    assert all(c.isalnum() or c in '/-_.:' for c in source)
    command = f"cat -- '{ROOT}/{source}'"
    event(event='download_started', path=item['path'], bytes=item['size'])
    process = subprocess.Popen(SSH + [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    total = 0
    last_report = time.monotonic()
    with destination.open('wb') as stream:
        while True:
            chunk = process.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            digest.update(chunk)
            total += len(chunk)
            if time.monotonic() - last_report >= 45:
                event(event='download_progress', path=item['path'], bytes=total, total=item['size'])
                last_report = time.monotonic()
    process.wait(timeout=30)
    assert process.returncode == 0 and total == item['size'] and digest.hexdigest() == item['sha256'], 'Server-to-local hash mismatch'
    event(event='download_verified', path=item['path'], bytes=total, sha256=digest.hexdigest())
    return destination


def small_files(package, manifest):
    slug = package['repo'].split('/')[1]
    folder = HERE / 'cards' / slug
    folder.mkdir(parents=True, exist_ok=True)
    public_manifest = dict(method=package['method'], source_code_commit=manifest['server_head'],
        selection=package['selection'], files=[{k:v for k,v in f.items() if k!='source'} for f in package['files']],
        scope='Self-trained inference parameters and necessary normalization only; no raw NSD data or frozen downloaded backbone weights.')
    (folder/'manifest.json').write_text(json.dumps(public_manifest,indent=2)+'\n',encoding='utf-8')
    (folder/'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in package['files']),encoding='utf-8')
    if slug=='neuroadapter-subject01-l450':
        raw=json.loads((HERE/'l450-metadata-source.json').read_text(encoding='utf-8-sig'))
        allowed=('run_id','seed','subject_id','topk','condition_dim','num_decoder_queries','sub_approach',
            'gen_img_resolution','learning_rate','weight_decay','clip_max_norm','mixed_precision',
            'train_batch_size','gradient_accumulation_steps','num_train_epochs')
        config={k:raw['config'][k] for k in allowed}
        config.update(base_model='stable-diffusion-v1-5/stable-diffusion-v1-5',num_parcels=raw['total_parcels'],max_voxels=raw['max_voxels_per_parcel'],
            world_size=2,trainable_state_keys=['guidance_generator','image_proj','ip_adapter'],epoch=450,
            saved_inference_sha256=package['files'][0]['sha256'])
        (folder/'inference-config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    # Copy only metric scalars, never per-sample records or server paths.
    metrics={}
    metric_names={'PixCorr','SSIM','Alex(2)','Alex(5)','Incep','CLIP','Eff','SwAV'}
    def metric_rows(value,prefix=''):
        if not isinstance(value,dict):return
        if metric_names.issubset(value):
            metrics[prefix]={k:value[k] for k in sorted(metric_names)}
            return
        for key,child in value.items():
            metric_rows(child,(prefix+'/'+key).strip('/'))
    metric_rows(package.get('metrics',{}))
    if metrics:
        (folder/'evaluation-summary.json').write_text(json.dumps(dict(n_test=982,metrics=metrics,
            notes='Recorded completed evaluations, no new generation. Identification metrics are percentages; other metrics are unitless.'),indent=2)+'\n',encoding='utf-8')
    notes = {
      'neuroadapter-subject01-l450': '''This is the validation-selected Subject 1 low-learning-rate epoch450 adapter, seed42, from the author-first reproduction. Training used frozen SD1.5, FP32, 200 parcels, linear mapper, dimension768, original TokenDropout and Min-SNR gamma5. LR schedule: epochs1-100 at1e-4,101-400 at5e-5,401-450 at1e-5; weight decay1e-6; global batch16. It is not the senior E750 model and not an exact reproduction of the paper's training budget.

`inference.pt` contains the trained adapter/mapper inference state; `parcel_vertices.pt` fixes the selected vertices. Use the matching author-first architecture and frozen SD1.5. Candidate0 K1 and brain-encoder best-of-eight use identical weights. Best8 chooses predicted/measured fMRI correlation, not GT-image similarity. Four identification metrics and Eff/SwAV improved while PixCorr/SSIM decreased. The test set has been viewed during development; no independent blind-test or significance claim.

Upstream: https://github.com/kriegeskorte-lab/NeuroAdapter ; paper: https://arxiv.org/abs/2509.23566 .''',
      'mindbridge-subject01-reproduction': '''Self-trained single-subject (vanilla) MindBridge, Subject1, seed42, 600epochs, two ranks, per-rank batch50, FP32. h_size2048, n_blocks4, max-pooling8192, max_lr3e-4, mse_mult10000, evaluation every10epochs, following the original train_single.sh recipe.

`last.pth` is the fixed600epoch endpoint used for the reported K1 and native8 results. `best.pth` is selected by native validation similarity and is included for preservation; the reported test metrics must not be attributed to it. No new training or seed search was performed for this release. In native8 evaluation, GT CLIP features participate in candidate selection. This is GT-assisted selection and is not comparable to a brain-only ranker under an identical inference-information budget. Frozen CLIP and Versatile Diffusion backbones must be obtained separately.

Upstream: https://github.com/littlepure2333/MindBridge .''',
      'takagi-subject01-regression': '''Two self-fitted Subject1 scikit-learn/himalaya pipelines, using original StandardScaler + RidgeCV: early visual ROI to initial latent (`init_latent.joblib`), and ventral ROI to text conditioning (`c.joblib`). These are fitted parameters, not downloaded SD weights. Training used24980 single trials and testing982 repeated-stimulus means with the recorded native voxel order. The original seven-alpha CV protocol chose regularization using training data.

The joblib pipelines include preprocessing and fitted regressors. Frozen SD1.4 must be obtained separately. The native five-candidate evaluation averages candidate feature correlations before computing two-way identification; it does not select one candidate. PixCorr/SSIM/Eff/SwAV are undefined in that native aggregation and are not claimed here. Library-version compatibility differs from the historical paper environment. No raw fMRI arrays, image features or generated images are included.

Upstream: https://github.com/yu-takagi/StableDiffusionReconstruction .''',
      'brain-diffuser-subject01-regression': '''Three self-fitted Subject1 regression heads from the original Brain-Diffuser recipe: VDVAE(alpha50000), CLIP vision(alpha60000) and CLIP text(alpha100000). The pretrained VDVAE/Versatile Diffusion/CLIP backbones are downloaded dependencies and are not republished.

The input is the native nsdgeneral ROI vector with15724voxels in the original C-order. Divide brain values by300, then apply stored training mean and std(ddof1). Each pickle contains `weight` and `bias`. CLIP weights retain the original author's float32 serialization; VDVAE retains its original serialization. CLIP target-normalization files contain only aggregate training-feature mean/std(ddof0), computed from the already-frozen training features, with no refitting.

After regression, the original pipeline z-scores predictions over the entire982test pool before applying training-target mean/std. This is a batch-dependent, transductive calibration: applying it independently to a smaller subset changes results. VDVAE normalization also preserves the original full982prediction statistics. Reconstructing from the serialized float32 CLIP coefficients can differ slightly from original in-memory regression predictions; no new bit-exact reconstruction claim is made. Frozen models, NSD inputs and per-sample target features are not part of this release.

Upstream: https://github.com/ozcelikfu/brain-diffuser .'''
    }
    readme = f"---\ntags:\n- fmri\n- nsd\n- reproduction\n- subject-1\n---\n\n# {package['method']} — Subject 1 reproduction weights\n\n{notes[slug]}\n\n## Provenance and integrity\n\nSource revision: `{manifest['server_head']}`. These repositories preserve our trained artifacts. `manifest.json` and `SHA256SUMS` list exact file sizes/hashes. This is an inference-weight release; full optimizer/RNG continuation states remain in the local project. The method name credits the original authors; this account is not the official upstream release. Use of upstream architectures and pretrained dependencies remains subject to their original licenses.\n"
    (folder/'README.md').write_text(readme,encoding='utf-8')
    return folder


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    credential = ssh_capture(f"cd {ROOT} && runtime/env/bin/python -B -c 'from huggingface_hub import get_token; print(get_token() or str())'")
    assert credential
    api = HfApi(token=credential)
    identity = api.whoami()
    assert identity['name'] == 'gugabobo'
    event(event='authenticated', account=identity['name'])
    receipts=[]
    for package in manifest['packages']:
        repo=package['repo']
        assert repo.startswith('gugabobo/')
        cards=small_files(package,manifest)
        api.create_repo(repo_id=repo,repo_type='model',private=False,exist_ok=True)
        assert api.model_info(repo).private is False
        operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(cards.iterdir())]
        commit=api.create_commit(repo_id=repo,operations=operations,commit_message='docs(weights): document trained model provenance and checksums')
        event(event='repository_ready',repo=repo,commit=commit.oid)
        for item in package['files']:
            if check_public(api,repo,item):
                event(event='already_verified',repo=repo,path=item['path'])
                continue
            local=download_one(item)
            event(event='upload_started',repo=repo,path=item['path'],bytes=item['size'])
            commit=api.upload_file(path_or_fileobj=str(local),path_in_repo=item['path'],repo_id=repo,
                commit_message='feat(weights): publish verified self-trained inference artifact')
            assert check_public(api,repo,item), 'Anonymous public SHA verification failed'
            event(event='public_sha_verified',repo=repo,path=item['path'],sha256=item['sha256'],commit=commit.oid)
            resolved=local.resolve()
            assert resolved.parent == SPOOL.resolve() and not local.is_symlink()
            local.unlink()
        info=api.model_info(repo,files_metadata=True,token=False)
        assert info.private is False
        for item in package['files']:
            assert check_public(api,repo,item)
        receipt=dict(repo=repo,url='https://huggingface.co/'+repo,revision=info.sha,public=True,
            file_count=len(package['files']),bytes=sum(f['size'] for f in package['files']),sha_verified=True)
        receipts.append(receipt)
        (HERE/'upload-result.partial.json').write_text(json.dumps(receipts,indent=2)+'\n')
        event(event='repository_complete',**receipt)
    result=dict(completed=True,unix_time=time.time(),repositories=receipts,all_public_and_sha_verified=True,
        server_weights_deleted=False,excluded=manifest['excluded'])
    (HERE/'upload-result.json').write_text(json.dumps(result,indent=2)+'\n')
    event(event='all_uploads_complete',repositories=len(receipts))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not echo exception URLs or request headers, which can contain signed credentials.
        import traceback
        frames=traceback.extract_tb(error.__traceback__)
        event(event='upload_failed',error_type=type(error).__name__,
              frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in frames[-4:]])
        raise SystemExit(1)
