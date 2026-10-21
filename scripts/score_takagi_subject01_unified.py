"""Rescore existing Takagi Subject1 candidates with the frozen common evaluator."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
B = R / "benchmarks/subj01-v1"
INPUT = B / "outputs/takagi-native5"
OUT = R / "reports/takagi-subject01-unified-metrics-20261021"
IDS = B / "test_full_ids.txt"
PY = R / "runtime/env/bin/python"
METRIC_SHA = "646ee1f3673b033cd6ee485058db16a3d24fae7624d793fe5174008ad2e15f7d"
BATCH_SHA = "08272975f6690413812cfb28b783faa9baf43ab02d1b88e921ea70b0a1822425"
KEYS = ["PixCorr", "SSIM", "Alex(2)", "Alex(5)", "Incep", "CLIP", "Eff", "SwAV"]
os.environ.update(TORCH_HOME=str(R / "runtime/torch"),
    AUTHOR_SWAV_SOURCE=str(R / "runtime/swav"),
    AUTHOR_CLIP_PATH=str(R.parent / "neuroadapter-subject1-research/models/evaluation/clip/ViT-L-14.pt"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
    CUBLAS_WORKSPACE_CONFIG=":4096:8")
sys.path[:0] = [str(R), str(B)]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write(name, value, fresh=False):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert path.resolve().is_relative_to(OUT.resolve())
    if fresh:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

def diskcheck():
    free = shutil.disk_usage(R).free / 2**30
    assert free >= 25, f"Disk reserve below 25GiB: {free}"
    return free

def frozen(all_inputs=False):
    plan = json.loads((OUT / "plan.json").read_text())
    for name, digest in plan["protected"].items():
        assert sha(Path(name)) == digest, name
    if all_inputs:
        for line in (OUT / "inputs.jsonl").read_text().splitlines():
            row = json.loads(line)
            for name, digest in row["files"].items():
                assert sha(R / name) == digest, name
    diskcheck()
    return plan

def prepare():
    import numpy as np
    from PIL import Image
    assert not OUT.exists(), "Existing report: inspect rather than overwrite."
    assert sha(R / "metric_brain_adapter.py") == METRIC_SHA
    assert sha(B / "metric_feature_batches.py") == BATCH_SHA
    ids = [int(x) for x in IDS.read_text().split()]
    assert len(ids) == len(set(ids)) == 982
    assert {int(p.stem) for p in INPUT.glob("*.json") if p.stem.isdigit()} == set(ids)
    assert len(list(INPUT.glob("*/samples/*.png"))) == 4910
    assert json.loads((R / "reports/native-protocols-20261016/takagi/generation-complete.json").read_text()) == {"n":982, "K":5}
    protected_paths = [Path(__file__), IDS, R / "metric_brain_adapter.py",
        B / "metric_feature_batches.py", B / "metric_protocol.json",
        INPUT / "protocol.json", B / "metrics/takagi/test_full_ids/metrics.json",
        B / "metrics/takagi/test_full_ids/per-image.csv",
        R / "reports/native-protocols-20261016/takagi/native-identification.json",
        Path(os.environ["AUTHOR_CLIP_PATH"])]
    protected_paths += list((R / "runtime/torch/hub/checkpoints").glob("*"))
    protected_paths += list((R / "runtime/swav").rglob("*.py"))
    protected = {str(p):sha(p) for p in protected_paths if p.is_file()}
    free = diskcheck()
    OUT.mkdir()
    write("plan.json", dict(subject=1, n=982, K=5, negative_pool=982,
        metric="metric_brain_adapter.py", metric_sha256=METRIC_SHA,
        feature_batch=16, aggregation="Score each fixed candidate index on all982 independently; arithmetic mean of the five metric results AFTER identification.",
        rows={"K1":"Candidate0 of the existing five", "K5-mean":"Mean of five separately evaluated candidate indices; no selection"},
        selection="none", candidate_indices=[0,1,2,3,4],
        GT="Existing original NSD425 RGB uint8/255",
        prediction="Existing native PNG uint8/255; author resize_images bilinear align_corners=False to425",
        training_updates=0, new_images=0, seed_changes=0,
        protected=protected, disk_reserve_gib=25,
        shared_gpu_authorization="User: 可以一起运行",
        resource_limits={"gpu":1,"torch_memory_fraction":0.28,"cpu_threads":4,"nice_increment":10},
        limitations=["This is a unified evaluator, not Takagi native five-correlation aggregation.",
                    "Candidate budgets and selectors of other methods remain different.",
                    "No test-based candidate, weight, seed or protocol selection."],
        created=time.time()), fresh=True)
    matched = []
    with (OUT / "inputs.jsonl").open("x") as out:
        for pos, i in enumerate(ids):
            rp = INPUT / f"{i}.json"
            rec = json.loads(rp.read_text())
            assert rec["nsd_id0"] == i and rec["K"] == 5 and rec["selection"] == "none"
            assert len(rec["candidates"]) == 5
            gp = B / "assets/gt" / f"{i}.png"
            oldp = B / "outputs/takagi" / f"{i}.png"
            files = {str(rp.relative_to(R)):sha(rp),str(gp.relative_to(R)):sha(gp),
                     str(oldp.relative_to(R)):sha(oldp)}
            with Image.open(gp) as im:
                assert im.mode == "RGB" and im.size == (425,425)
            for k, c in enumerate(rec["candidates"]):
                p = (INPUT / c["path"]).resolve()
                assert p.is_relative_to(INPUT.resolve()) and p.name.endswith(f"_{k:03}.png")
                digest = sha(p)
                assert digest == c["sha256"]
                files[str(p.relative_to(R))] = digest
                with Image.open(p) as im:
                    assert im.mode == "RGB" and im.size == (512,512)
                    arr = np.asarray(im)
                    assert arr.dtype == np.uint8 and arr.std() > 0
                    if k == 0:
                        with Image.open(oldp) as old:
                            if np.array_equal(arr, np.asarray(old.convert("RGB"))):
                                matched.append(i)
            out.write(json.dumps(dict(image_id=i,files=files),sort_keys=True)+"\n")
            if (pos+1)%100 == 0:
                print("INPUT_AUDIT",pos+1,flush=True)
    write("input-audit.json",dict(n=982,candidates=4910,all_record_file_hashes_verified=True,
        all_RGB_shapes_valid=True,old_K1_pixel_identical_count=len(matched),
        old_K1_all_pixels_identical=len(matched)==982,free_gib=free,
        inputs_sha256=sha(OUT/"inputs.jsonl")),fresh=True)
    print("INPUTS_VERIFIED",len(matched),"old K1 matches",flush=True)

def score(k):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    import metric_brain_adapter as m
    from metric_feature_batches import install
    plan = frozen()
    assert sha(OUT/"inputs.jsonl") == json.loads((OUT/"input-audit.json").read_text())["inputs_sha256"]
    assert k in plan["candidate_indices"]
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.28)
    install(m,16)
    ids = [int(x) for x in IDS.read_text().split()]
    gt, pred = [], []
    for i in ids:
        rec = json.loads((INPUT/f"{i}.json").read_text())
        path = INPUT/rec["candidates"][k]["path"]
        assert sha(path) == rec["candidates"][k]["sha256"]
        with Image.open(B/"assets/gt"/f"{i}.png") as im:
            gt.append(torch.from_numpy(np.array(im)).permute(2,0,1).float()/255)
        with Image.open(path) as im:
            pred.append(torch.from_numpy(np.array(im)).permute(2,0,1).float()/255)
    originals = torch.stack(gt)
    images = m.resize_images(torch.stack(pred), originals.shape[2:])
    del gt,pred
    with torch.no_grad():
        frame, summary = m.MetricEvaluator(device="cuda").compute_all_metrics(originals,images)
    assert list(frame.columns) == KEYS and frame.shape == (982,8)
    assert np.isfinite(frame.to_numpy()).all()
    frame.insert(0,"image_id",ids)
    frame.to_csv(OUT/f"candidate-{k}-per-image.csv",index=False,mode="x")
    means = summary["mean"].to_dict()
    for key in KEYS:
        assert abs(frame[key].mean()-means[key]) < 1e-10
    write(f"candidate-{k}-metrics.json",dict(candidate=k,n=982,negative_pool=982,mean=means,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        runtime=dict(torch=torch.__version__,device=torch.cuda.get_device_name(),
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32)),fresh=True)
    print("CANDIDATE_COMPLETE",k,flush=True)

def aggregate():
    import numpy as np
    import pandas as pd
    plan = frozen(all_inputs=True)
    ids = [int(x) for x in IDS.read_text().split()]
    frames, summaries = [], []
    for k in range(5):
        assert json.loads((OUT/f"candidate-{k}-exit.json").read_text())["returncode"] == 0
        log = (OUT/f"candidate-{k}.log").read_text()
        assert "SSIM computation failed" not in log
        frame = pd.read_csv(OUT/f"candidate-{k}-per-image.csv")
        assert frame.image_id.tolist() == ids
        assert np.isfinite(frame[KEYS].to_numpy()).all()
        metrics = json.loads((OUT/f"candidate-{k}-metrics.json").read_text())
        for key in KEYS:
            assert abs(frame[key].mean()-metrics["mean"][key]) < 1e-10
        frames.append(frame[KEYS].to_numpy())
        summaries.append(metrics["mean"])
    averaged = np.stack(frames).mean(axis=0)
    mean_frame = pd.DataFrame(averaged,columns=KEYS)
    mean_frame.insert(0,"image_id",ids)
    mean_frame.to_csv(OUT/"K5-mean-per-image.csv",index=False,mode="x")
    means = {key:float(averaged[:,j].mean()) for j,key in enumerate(KEYS)}
    for key in KEYS:
        assert abs(means[key]-np.mean([s[key] for s in summaries])) < 1e-10
    old = json.loads((B/"metrics/takagi/test_full_ids/metrics.json").read_text())["mean"]
    result = dict(subject=1,n=982,negative_pool=982,metric_sha256=METRIC_SHA,
        feature_batch=16,selection="none",K1=summaries[0],K5_mean=means,
        per_candidate=summaries,aggregation=plan["aggregation"],
        K1_minus_historical={key:summaries[0][key]-old[key] for key in KEYS},
        training_updates=0,new_images=0,completed=time.time())
    write("metrics.json",result,fresh=True)
    write("final-audit.json",dict(complete=True,all_input_hashes_unchanged=True,
        all_protected_evaluator_and_historical_files_unchanged=True,
        n=982,candidates=4910,per_candidate_full_negative_pool=982,
        candidate_and_aggregate_means_verified=True,SSIM_exception_count=0,
        historical_K1_pixel_comparison=json.loads((OUT/"input-audit.json").read_text()),
        new_image_or_candidate_selection=False,
        results_sha256={p.name:sha(p) for p in OUT.glob("*per-image.csv")}),fresh=True)

def controller():
    assert (OUT/"input-audit.json").exists()
    with (OUT/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/"started.json").exists(), "No automatic restart."
        frozen()
        os.nice(10)
        write("started.json",dict(pid=os.getpid(),time=time.time(),shared_gpu_authorized=True),fresh=True)
        for k in range(5):
            free_mem = int(subprocess.check_output(["nvidia-smi","--id=1",
                "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
            assert free_mem >= 9000, "Insufficient free GPU memory; preserve state and inspect."
            write("status.json",dict(phase="scoring",candidate=k,total_candidates=5,pid=os.getpid(),gpu=1,time=time.time()))
            env = dict(os.environ,CUDA_VISIBLE_DEVICES="1",PYTHONUNBUFFERED="1")
            with (OUT/f"candidate-{k}.log").open("x") as log:
                p = subprocess.Popen([str(PY),"-B",str(Path(__file__)),"score","--candidate",str(k)],
                    cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
                write(f"candidate-{k}-started.json",dict(pid=p.pid,time=time.time(),gpu=1),fresh=True)
                code = p.wait()
            write(f"candidate-{k}-exit.json",dict(returncode=code,time=time.time()),fresh=True)
            assert code == 0, f"Candidate{k} failed; inspect retained log."
        aggregate()
        write("status.json",dict(phase="complete",time=time.time(),n=982,candidates=5))
        write("exit.json",dict(returncode=0,time=time.time()),fresh=True)
        print("UNIFIED_METRICS_COMPLETE",flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","controller","score"])
    p.add_argument("--candidate",type=int)
    args=p.parse_args()
    try:
        {"prepare":prepare,"controller":controller,"score":lambda:score(args.candidate)}[args.mode]()
    except BaseException:
        if OUT.exists():
            write(f"failure-{args.mode}-{args.candidate}.json",dict(traceback=traceback.format_exc(),time=time.time()))
        raise
