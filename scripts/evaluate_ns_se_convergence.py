#!/usr/bin/env python3
"""
NS SE convergence: find minimum K to reach capacity SE at each dimension.
Compares with BJ at same dimensions.
"""
import numpy as np, sys, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_ldl_quality import EvalConfig, estimate_se, quantize_complex
from scripts.reconstruct_formula_se_compare import (
    build_block_richardson_preconditioner, chebyshev_omega_adaptive,
)

np.random.seed(42)


def ns_inverse(a_mat, iters=5):
    """Newton-Schulz: X0 = alpha*A^T, X_{k+1} = X_k(2I - A X_k)."""
    n = a_mat.shape[0]
    real_a = np.abs(a_mat)
    norm1 = np.max(np.sum(real_a, axis=0))
    norm_inf = np.max(np.sum(real_a, axis=1))
    alpha = 1.0 / max(norm1 * norm_inf, 1e-12)
    a = a_mat.astype(np.complex64)
    x = (alpha * a.conj().T).astype(np.complex64)
    i2 = (2.0 * np.eye(n)).astype(np.complex64)
    for _ in range(iters):
        x = x @ (i2 - a @ x)
    return x


def ns_se(h, noise_var, K, ecfg):
    """SE for NS at K iterations."""
    nt = h.shape[1]
    a = h.conj().T @ h + noise_var * np.eye(nt)
    a = quantize_complex(a, ecfg)
    ai = ns_inverse(a, iters=K)
    ai = quantize_complex(ai, ecfg)
    w = quantize_complex(ai @ h.conj().T, ecfg)
    return estimate_se(w, h, noise_var, "64qam")


def bj_se_ref(h, noise_var, B, L, ecfg):
    """SE for BJ at given B, L."""
    nt = h.shape[1]
    a = quantize_complex(h.conj().T @ h + noise_var * np.eye(nt), ecfg)
    bm, bi = build_block_richardson_preconditioner(a, blk=B,
        precond_solver="direct2x2" if B == 2 else "cholesky")
    y = np.zeros_like(a, dtype=complex)
    I = np.eye(nt, dtype=complex)
    for w in chebyshev_omega_adaptive(bm, L, nt=nt):
        y = quantize_complex(y + w * (I - bm @ y), ecfg)
    ai = quantize_complex(y @ bi, ecfg)
    return estimate_se(quantize_complex(ai @ h.conj().T, ecfg), h, noise_var, "64qam")


def ec(nr, nt):
    return EvalConfig(nr=nr, nt=nt, n_sc=1, batch=1, trials=1, block_size=2,
        snr_db_list=[0.0], channel_model="rayleigh", pilot_len=nt, pilot_snr_db=None,
        num_format="float16", reciprocal_mode="lut", trunc_mantissa_bits=10,
        modulation="64qam", mac_chunk=1, seed=42, out_dir="/tmp")


def main():
    dims = [(64, 16, 16, [2, 8, 16]), (128, 32, 32, [2, 8, 16, 32]), (256, 64, 64, [2, 8, 16, 32, 64])]
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    K_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
    trials = 5

    all_rows = []
    t0 = time.time()

    for nr, nt, U, Bs in dims:
        cap = U * 6
        print(f"\n{'='*70}")
        print(f"U={U} ({nr}x{nt}), cap={cap}")
        print(f"{'='*70}")

        for snr_db in snr_list:
            noise_var = 10**(-snr_db/10)
            ns_se_k = {K: [] for K in K_list}
            bj_se_data = {}

            for t in range(trials):
                h = (np.random.randn(nr, nt) + 1j * np.random.randn(nr, nt)) / np.sqrt(2)

                # NS at all K
                for K in K_list:
                    ns_se_k[K].append(ns_se(h, noise_var, K, ec(nr, nt)))

                # BJ reference at key (B, L) combos - only need once per SNR
                if t == 0:
                    for B in Bs:
                        bj_se_data[B] = {}
                        for L in [1, 2, 3, 4, 6, 8, 12]:
                            try:
                                bj_se_data[B][L] = bj_se_ref(h, noise_var, B, L, ec(nr, nt))
                            except Exception as e:
                                bj_se_data[B][L] = -1

            # Print NS results and find knee
            ns_means = {K: float(np.mean(v)) for K, v in ns_se_k.items()}
            min_K_cap = None
            for K in sorted(K_list):
                if ns_means[K] >= cap - 0.5:
                    min_K_cap = K
                    break

            ns_str = "  ".join(f"K={K}:{ns_means[K]:.1f}" for K in [1,2,3,4,5,8,10,14,20] if K in ns_means)
            print(f"  SNR={snr_db:2d}dB | {ns_str}")
            if min_K_cap:
                print(f"    NS reaches cap-0.5 at K={min_K_cap} (SE={ns_means[min_K_cap]:.1f})")

            # Print BJ results
            if snr_db == 20:
                print(f"  BJ@20dB:")
                for B in Bs:
                    best = sorted([(L, s) for L, s in bj_se_data.get(B, {}).items() if s >= 0],
                                  key=lambda x: x[1], reverse=True)
                    if best:
                        L_best, se_best = best[0]
                        print(f"    B={B}: best L={L_best} SE={se_best:.1f}")

            # Record data
            for K, v in ns_means.items():
                all_rows.append({"U": U, "method": "NS", "param": f"K={K}",
                                 "snr": snr_db, "se": round(v, 2),
                                 "cap": cap, "reaches_cap": v >= cap - 0.5})
            for B, d in bj_se_data.items():
                for L, v in d.items():
                    if v >= 0:
                        all_rows.append({"U": U, "method": "BJ", "param": f"B={B},L={L}",
                                         "snr": snr_db, "se": round(v, 2),
                                         "cap": cap, "reaches_cap": v >= cap - 0.5})

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY: Minimum K/L to reach capacity SE at SNR=20dB")
    print(f"{'='*90}")
    for nr, nt, U, Bs in dims:
        cap = U * 6
        rows_u = [r for r in all_rows if r["U"] == U and r["snr"] == 20]

        # NS min K
        ns_rows = sorted([r for r in rows_u if r["method"] == "NS"], key=lambda r: int(r["param"].split("=")[1]))
        min_k = None
        for r in ns_rows:
            if r["reaches_cap"]:
                min_k = int(r["param"].split("=")[1])
                break
        ns_se_val = next((r["se"] for r in ns_rows if int(r["param"].split("=")[1]) == min_k), "?") if min_k else "?"

        # BJ best
        bj_rows = [r for r in rows_u if r["method"] == "BJ"]
        bj_best = sorted(bj_rows, key=lambda r: r["se"], reverse=True)
        bj_top = bj_best[0] if bj_best else {"param": "?", "se": "?"}

        print(f"U={U} (cap={cap}): NS min K={min_k} (SE={ns_se_val}), BJ best {bj_top['param']} (SE={bj_top['se']})")

    # Save
    out = "/project/Asim/result_new/ns_se_convergence.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["U", "method", "param", "snr", "se", "cap", "reaches_cap"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows → {out}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
