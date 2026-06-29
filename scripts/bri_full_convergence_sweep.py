#!/usr/bin/env python3
"""
Complete BJ convergence sweep: all dimensions × all block sizes × key channel conditions.
Produces optimal-layers table for report.
"""
import numpy as np, sys, csv, time
sys.path.insert(0, "/project/Asim")
from scripts.evaluate_ldl_quality import EvalConfig, estimate_se, quantize_complex
from scripts.reconstruct_formula_se_compare import (
    build_block_richardson_preconditioner, chebyshev_omega_adaptive,
)
np.random.seed(42)

def ec(nr, nt):
    return EvalConfig(nr=nr, nt=nt, n_sc=1, batch=1, trials=1, block_size=2,
        snr_db_list=[0.0], channel_model="rayleigh", pilot_len=nt, pilot_snr_db=None,
        num_format="float16", reciprocal_mode="lut", trunc_mantissa_bits=10,
        modulation="64qam", mac_chunk=1, seed=42, out_dir="/tmp")

# ── Channel generators ────────────────────────────────────────────────
def ch_iid(nr, nt):
    return (np.random.randn(nr, nt) + 1j*np.random.randn(nr, nt)) / np.sqrt(2)

def ch_correlated(nr, nt, rho):
    h = ch_iid(nr, nt)
    idx = np.arange(nt)
    C = rho ** np.abs(idx[:, None] - idx[None, :])
    return h @ np.linalg.cholesky(C + 1e-10*np.eye(nt)).T

def ch_los(nr, nt, k_db):
    k = 10**(k_db/10)
    h = ch_iid(nr, nt)
    los = np.ones((nr, nt), dtype=complex) / np.sqrt(nr*nt)
    return np.sqrt(k/(k+1))*los*np.sqrt(nr*nt) + np.sqrt(1/(k+1))*h

CHANNELS = {
    "iid":         (lambda nr,nt: ch_iid(nr, nt), "i.i.d. Rayleigh"),
    "corr_ρ0.6":   (lambda nr,nt: ch_correlated(nr, nt, 0.6), "TX corr ρ=0.6"),
    "corr_ρ0.8":   (lambda nr,nt: ch_correlated(nr, nt, 0.8), "TX corr ρ=0.8"),
    "LOS_K6dB":    (lambda nr,nt: ch_los(nr, nt, 6), "Rician K=6dB"),
}

LAYERS = [1,2,3,4,6,8,10,12,16,20,24,32,40,48]

def bj_se(h, noise_var, B, L, ecfg):
    nt = h.shape[1]
    a = quantize_complex(h.conj().T @ h + noise_var*np.eye(nt), ecfg)
    bm, bi = build_block_richardson_preconditioner(a, blk=B,
        precond_solver="direct2x2" if B==2 else "cholesky")
    y = np.zeros_like(a, dtype=complex)
    I = np.eye(nt, dtype=complex)
    for w in chebyshev_omega_adaptive(bm, L, nt=nt):
        y = quantize_complex(y + w*(I - bm @ y), ecfg)
    ai = quantize_complex(y @ bi, ecfg)
    return estimate_se(quantize_complex(ai @ h.conj().T, ecfg), h, noise_var, "64qam")

def sweep_dim(nr, nt, U, B_list, snr=20, trials=3):
    cap = U * 6
    noise_var = 10**(-snr/10)
    all_rows = []
    for ch_name, (ch_fn, ch_desc) in CHANNELS.items():
        # Compute DD metrics
        h0 = ch_fn(nr, nt)
        a0 = h0.conj().T @ h0 + noise_var*np.eye(nt)
        d = np.abs(np.diag(a0))
        o = np.sum(np.abs(a0), axis=1) - d
        dd = float(np.min(d / (o + 1e-12)))
        cond = float(np.linalg.cond(a0))

        for B in B_list:
            print(f"    {ch_name} B={B}: ", end="", flush=True)
            for L in LAYERS:
                se_vals = []
                for t in range(trials):
                    h = ch_fn(nr, nt)
                    se_vals.append(bj_se(h, noise_var, B, L, ec(nr, nt)))
                avg = float(np.mean(se_vals))
                all_rows.append({"U":U, "B":B, "channel":ch_name, "desc":ch_desc,
                    "layers":L, "se":round(avg,2), "min_dd":round(dd,4), "cond":round(cond,1)})
            print("done")
    return all_rows

# ══════════════════════════════════════════════════════════════════════
ALL = []
t0 = time.time()

for nr, nt, U, Bs, tr in [
    (64, 16, 16, [2,4,8,16], 4),
    (128, 32, 32, [2,4,8,16,32], 3),
    (256, 64, 64, [2,4,8,16,32,64], 2),
]:
    print(f"\n{'='*60}")
    print(f"Sweep: U={U} ({nr}×{nt}), trials={tr}")
    print(f"{'='*60}")
    ALL.extend(sweep_dim(nr, nt, U, Bs, 20, tr))

print(f"\nTotal time: {time.time()-t0:.0f}s")

# Save raw data
FIELDS = ["U","B","channel","desc","layers","se","min_dd","cond"]
with open("/project/Asim/result_new/bj_full_sweep.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader(); w.writerows(ALL)
print(f"Saved: {len(ALL)} points → result_new/bj_full_sweep.csv")

# ── Compute knee table ────────────────────────────────────────────────
print("\n" + "="*90)
print("OPTIMAL LAYERS TABLE (min layers to reach SE≥cap−0.5)")
print("="*90)
for dim_name, U in [("U=16 (64×16, cap=96)",16),("U=32 (128×32, cap=192)",32),("U=64 (256×64, cap=384)",64)]:
    print(f"\n{dim_name}:")
    header = f"{'Channel':<15} {'min_DD':<8} {'cond':<8} " + \
             "".join(f"B={b:<5}" for b in ([2,4,8,16] if U==16 else [2,4,8,16,32] if U==32 else [2,4,8,16,32,64]))
    print(header)
    print("-"*len(header))
    for ch_name, (_, ch_desc) in CHANNELS.items():
        rows = [r for r in ALL if r["U"]==U and r["channel"]==ch_name]
        dd = rows[0]["min_dd"] if rows else 0
        cond = rows[0]["cond"] if rows else 0
        Bs = sorted(set(r["B"] for r in rows))
        knee_str = ""
        for B in Bs:
            br = sorted([r for r in rows if r["B"]==B], key=lambda x: x["layers"])
            k = max(r["layers"] for r in br)
            for r in br:
                cap = U * 6
                if r["se"] >= cap - 0.5:
                    k = r["layers"]; break
            knee_str += f" {k:<5}"
        print(f"{ch_name:<15} {dd:<8.3f} {cond:<8.0f}{knee_str}")
