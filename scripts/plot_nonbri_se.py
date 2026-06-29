#!/usr/bin/env python3
"""
Plot individual SE-SNR curves for Cholesky-Block, Cholesky-NoBlock, LDL-Block, LDL-NoBlock.
Data extracted from the verified document tables (algorithmically identical for all exact-inverse methods,
but plotted individually for documentation completeness).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

snr = [0, 5, 10, 15, 20, 25, 30]
caps = {16: 96, 32: 192, 64: 384}

# SE data from document tables (all exact-inverse methods produce near-identical SE)
se_data = {
    "Cholesky-Block": {
        16: [31.799, 55.096, 80.251, 96.000, 96.000, 96.000, 96.000],
        32: [63.938, 109.109, 159.278, 192.000, 192.000, 192.000, 192.000],
        64: [128.223, 217.551, 317.929, 384.000, 384.000, 384.000, 384.000],
    },
    "Cholesky-NoBlock": {
        16: [31.799, 55.096, 80.251, 96.000, 96.000, 96.000, 96.000],
        32: [63.938, 109.109, 159.278, 192.000, 192.000, 192.000, 192.000],
        64: [128.223, 217.551, 317.929, 384.000, 384.000, 384.000, 384.000],
    },
    "LDL-Block": {
        16: [31.798, 55.097, 80.251, 96.000, 96.000, 96.000, 96.000],
        32: [63.939, 109.108, 159.277, 192.000, 192.000, 192.000, 192.000],
        64: [128.222, 217.551, 317.931, 384.000, 384.000, 384.000, 384.000],
    },
    "LDL-NoBlock": {
        16: [31.798, 55.097, 80.252, 96.000, 96.000, 96.000, 96.000],
        32: [63.939, 109.108, 159.277, 192.000, 192.000, 192.000, 192.000],
        64: [128.221, 217.552, 317.932, 384.000, 384.000, 384.000, 384.000],
    },
}

colors = {16: "#2196F3", 32: "#FF9800", 64: "#4CAF50"}
markers = {16: "o", 32: "s", 64: "^"}
labels = {16: "U=16 (64×16)", 32: "U=32 (128×32)", 64: "U=64 (256×64)"}

out_dir = Path("/project/Asim/DOCS/pic")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

# ── Individual algorithm plots ──────────────────────────────────────
for alg_name, alg_data in se_data.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    fname = alg_name.lower().replace("-", "_").replace(" ", "_")

    for u in [16, 32, 64]:
        ax.plot(snr, alg_data[u], marker=markers[u], color=colors[u],
                linewidth=1.8, markersize=6, label=labels[u])
        ax.axhline(y=caps[u], color=colors[u], linestyle="--", alpha=0.3,
                   linewidth=1)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SE (bit/s/Hz)")
    ax.set_title(f"{alg_name} SE-SNR (64QAM, Rayleigh+LS+FP16)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    ax.set_xlim(-1, 31)
    ax.set_ylim(0, 410)

    fig.tight_layout()
    path = out_dir / f"se_{fname}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path}")

# ── 2×2 grid: all non-BJ algorithms together ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
algs = list(se_data.keys())

for idx, (ax, alg_name) in enumerate(zip(axes.flat, algs)):
    alg_data = se_data[alg_name]
    for u in [16, 32, 64]:
        ax.plot(snr, alg_data[u], marker=markers[u], color=colors[u],
                linewidth=1.8, markersize=6, label=labels[u])
        ax.axhline(y=caps[u], color=colors[u], linestyle="--", alpha=0.3,
                   linewidth=1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SE (bit/s/Hz)")
    ax.set_title(alg_name, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-1, 31)
    ax.set_ylim(0, 410)

fig.suptitle("Exact-Inverse SE-SNR (64QAM, Rayleigh+LS+FP16, Batch=96)",
             fontsize=14, y=1.01)
fig.tight_layout()
path = out_dir / "nonbj_se_grid.png"
fig.savefig(path, dpi=180)
plt.close(fig)
print(f"Saved: {path}")
