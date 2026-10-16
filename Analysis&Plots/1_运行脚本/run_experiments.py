# -*- coding: utf-8 -*-
"""实验驱动：串行跑多种子/消融/敏感性实验，解析最终 mse/mae 汇总到 CSV。"""
import subprocess, sys, re, csv, os, time, io

PY = r"C:\Users\马加帅\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
CWD = r"D:\bisai"
RUN = r"LLMNet-main/LLMNet-main/run.py"
OUT = r"D:\bisai\experiment_results.csv"
LOG = r"D:\bisai\experiment_log.txt"

def run_one(tag, extra):
    cmd = [PY, RUN, "--data", "Agricultural", "--model", "TriAlignNet",
           "--root_path", "./Dataset/Dataset", "--data_path", "protein_output.csv",
           "--is_training", "1", "--des", tag] + extra
    t0 = time.time()
    p = subprocess.run(cmd, cwd=CWD, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    txt = (p.stdout or "") + (p.stderr or "")
    ms = re.findall(r"mse:([\d.eE+-]+), mae:([\d.eE+-]+)", txt)
    if ms:
        mse, mae = ms[-1]
        print(f"[OK] {tag} {' '.join(extra)} -> MSE={mse} MAE={mae} ({dt:.0f}s)", flush=True)
        return mse, mae
    print(f"[FAIL] {tag} {' '.join(extra)} ({dt:.0f}s)", flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {tag} {' '.join(extra)} =====\n{txt[-3000:]}\n")
    return None, None

jobs = []
# 1) 多种子主实验（完整模型）
for seed in (1110, 2222, 3333):
    for pl in (6, 12, 18, 24):
        jobs.append((f"full_seed{seed}_pl{pl}", ["--seq_len", "24", "--pred_len", str(pl), "--seed", str(seed)]))
# 2) 消融（seed 1110）
for flag, name in (("--disable_mmd", "no_mmd"), ("--disable_anchor", "no_anchor"), ("--disable_fusion", "no_fusion")):
    for pl in (6, 12, 18, 24):
        jobs.append((f"{name}_pl{pl}", ["--seq_len", "24", "--pred_len", str(pl), flag]))
# 3) 敏感性（seed 1110）
for sl in (36, 48):
    for pl in (6, 12, 18, 24):
        jobs.append((f"seqlen{sl}_pl{pl}", ["--seq_len", str(sl), "--pred_len", str(pl)]))
for dm in (128, 256):
    for pl in (6, 12, 18, 24):
        jobs.append((f"dmodel{dm}_pl{pl}", ["--seq_len", "24", "--pred_len", str(pl), "--d_model", str(dm), "--d_ff", str(dm)]))

with io.open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tag", "seq_len", "pred_len", "seed", "variant", "MSE", "MAE"])
    for tag, extra in jobs:
        mse, mae = run_one(tag, extra)
        # 解析参数
        d = dict(zip(extra[::2], extra[1::2]))
        seed = d.get("--seed", "1110")
        variant = "full"
        if "--disable_mmd" in d: variant = "no_mmd"
        elif "--disable_anchor" in d: variant = "no_anchor"
        elif "--disable_fusion" in d: variant = "no_fusion"
        elif "--seq_len" in d and d["--seq_len"] != "24": variant = f"seq{d['--seq_len']}"
        elif "--d_model" in d: variant = f"dm{d['--d_model']}"
        w.writerow([tag, d.get("--seq_len", "24"), d.get("--pred_len"), seed, variant, mse, mae])
        f.flush()
print("ALL DONE ->", OUT, flush=True)
