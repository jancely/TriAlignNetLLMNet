# -*- coding: utf-8 -*-
"""生成 图5-3 跨模态注意力热力图 与 图5-4 模态分布 t-SNE 对比图。
思路：以与主实验完全相同的参数在进程内训练+测试（seed1110, pl12），
测试阶段通过 monkey-patch 捕获 GlobalAttentionLayer 注意力矩阵与
Multimodal_alignment 对齐后的双模态特征，再绘图。
"""
import os, sys, runpy

ROOT = r"D:\bisai"
PRJ = os.path.join(ROOT, "LLMNet-main", "LLMNet-main")
OUTDIR = r"C:\Users\马加帅\WorkBuddy\2026-10-10-14-37-48\docx_check"
os.chdir(ROOT)
sys.path.insert(0, PRJ)

import torch
import torch.nn.functional as F
import numpy as np

COLLECT = {"on": False}
ATT = []            # 每个测试 batch 的注意力 (N, ql, kl)，多头平均
ENERGY = []         # softmax 前的余弦相似度能量 (N, ql, kl)，多头平均
FEATS = {"text": [], "time": []}   # 对齐后的特征

# ---------- patch 1: 捕获注意力矩阵 ----------
import layers.multimodal_Transformer as MT

_orig_attn_fwd = MT.GlobalAttentionLayer.forward

def _attn_fwd(self, query, values, keys, epoch, mask=None):
    out = _orig_attn_fwd(self, query, values, keys, epoch, mask)
    if COLLECT["on"] and not self.training:
        with torch.no_grad():
            N = query.shape[0]
            vl, ql = values.shape[1], query.shape[1]
            v = self.values(values)
            k = self.keys(keys)
            q = self.queries(query)
            v = v.reshape(N, vl, self.heads, self.head_dim)
            k = k.reshape(N, vl, self.heads, self.head_dim)
            q = q.reshape(N, ql, self.heads, self.head_dim)
            qn = F.normalize(q, p=2, dim=-1)
            kn = F.normalize(k, p=2, dim=-1)
            energy = torch.einsum("nqhd,nkhd->nhqk", [qn, kn])
            ENERGY.append(energy.mean(1).detach().cpu().numpy())  # N,ql,kl
            attn = F.softmax(energy / (self.head_dim ** 0.5), dim=3)  # N,h,ql,kl
            attn = attn.mean(1).detach().cpu().numpy()  # N,ql,kl
            ATT.append(attn)
    return out

MT.GlobalAttentionLayer.forward = _attn_fwd

# ---------- patch 2: 捕获对齐后的双模态特征 ----------
import layers.Multimodal_interaction as MI

_orig_align_fwd = MI.Multimodal_alignment.forward

def _align_fwd(self, x_text, x_time):
    res = _orig_align_fwd(self, x_text, x_time)
    if COLLECT["on"] and not self.training:
        FEATS["text"].append(res[0].detach().cpu())
        FEATS["time"].append(res[1].detach().cpu())
    return res

MI.Multimodal_alignment.forward = _align_fwd

# ---------- patch 3: 仅在 exp.test 阶段开启采集 ----------
import exp.exp_long_term_forecasting as ELF

_orig_test = ELF.Exp_Long_Term_Forecast.test

def _test(self, setting, test=0):
    COLLECT["on"] = True
    try:
        return _orig_test(self, setting, test)
    finally:
        COLLECT["on"] = False

ELF.Exp_Long_Term_Forecast.test = _test

# ---------- 以主实验参数运行 ----------
sys.argv = [
    "run.py",
    "--data", "Agricultural", "--model", "TriAlignNet",
    "--root_path", "./Dataset/Dataset", "--data_path", "protein_output.csv",
    "--is_training", "1", "--des", "figgen",
    "--seq_len", "24", "--pred_len", "12", "--seed", "1110",
]
runpy.run_path(os.path.join(PRJ, "run.py"), run_name="__main__")

# ---------- 绘图 ----------
np.savez(os.path.join(OUTDIR, "fig_captures.npz"),
         att=np.concatenate(ATT, axis=0),
         energy=np.concatenate(ENERGY, axis=0),
         ft=torch.cat(FEATS["text"], dim=0).numpy(),
         tm=torch.cat(FEATS["time"], dim=0).numpy())
print("captures saved")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 图 5-3：跨模态注意力热力图（基于 softmax 前余弦相似度能量，反映查询-键语义相关结构）
energy = np.concatenate(ENERGY, axis=0)      # (M, ql, kl)
print("energy captured:", energy.shape, "min/max:", energy.min(), energy.max())
# 展示代表性单样本窗口（第 1 个测试窗口），保留真实结构差异
emap = energy[0]
# 平滑：对测试窗口取均值也会平滑掉结构；此处用窗口均值+单样本的折中：取逐窗口标准化后的均值
e_z = (energy - energy.mean(axis=(1, 2), keepdims=True)) / (energy.std(axis=(1, 2), keepdims=True) + 1e-8)
emap = e_z.mean(axis=0)   # 平均相对注意力结构（相对每个窗口自身均值的偏离）
print("emap range:", emap.min(), emap.max())
ql, kl = emap.shape
half = kl // 2
fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
divnorm = matplotlib.colors.TwoSlopeNorm(vmin=emap.min(), vcenter=0.0, vmax=emap.max())
im = ax.imshow(emap, cmap="RdBu_r", norm=divnorm, aspect="auto")
ax.set_xticks(range(kl))
ax.set_xticklabels([f"锚点{i+1}" for i in range(half)] + [f"T{i+1}" for i in range(kl-half)], fontsize=8, rotation=45)
ax.set_yticks(range(ql))
ax.set_yticklabels([f"Q{i+1}" for i in range(ql)], fontsize=9)
ax.set_xlabel("键（Key）：时间锚点表示 + 文本语义通道", fontsize=11)
ax.set_ylabel("查询（Query）：时间模态表示", fontsize=11)
ax.axvline(half - 0.5, color="#444444", lw=1.4, ls="--")
ax.text(half*0.5, -1.35, "时间锚点键", ha="center", fontsize=9, color="#8B2E2E")
ax.text(half + (kl-half)*0.5, -1.35, "文本通道键", ha="center", fontsize=9, color="#8B2E2E")
for i in range(ql):
    for j in range(kl):
        v = emap[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(v) > 0.72*max(abs(emap.min()), abs(emap.max())) else "#333333")
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("归一化注意力得分（相对窗口均值的偏离，多头平均）", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig5_3_attention_heatmap.png"), dpi=200)
plt.close(fig)

# 图 5-4：模态分布 t-SNE 对比图
from sklearn.manifold import TSNE

def z_score(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

ft = torch.cat(FEATS["text"], dim=0)   # (M, 12, d)
tm = torch.cat(FEATS["time"], dim=0)   # (M, 12, d)
ft = ft.reshape(-1, ft.shape[-1]).numpy()   # (M*12, d)
tm = tm.reshape(-1, tm.shape[-1]).numpy()
print("text feats:", ft.shape, "time feats:", tm.shape)

rng = np.random.RandomState(0)
n_text, n_time = ft.shape[0], tm.shape[0]
idx_t = rng.choice(n_text, min(600, n_text), replace=False)
idx_s = rng.choice(n_time, min(600, n_time), replace=False)
X = np.concatenate([z_score(ft[idx_t]), z_score(tm[idx_s])], axis=0)
Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
n1 = len(idx_t)
fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=200)
ax.scatter(Z[:n1, 0], Z[:n1, 1], s=14, c="#E07B39", alpha=0.65, label="文本模态（对齐后）", edgecolors="none")
ax.scatter(Z[n1:, 0], Z[n1:, 1], s=14, c="#2E6F8E", alpha=0.65, label="时序模态（对齐后）", edgecolors="none")
ax.legend(fontsize=11, loc="best", frameon=True)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("对齐后双模态特征分布（t-SNE）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig5_4_tsne_modal.png"), dpi=200)
plt.close(fig)
print("DONE")
