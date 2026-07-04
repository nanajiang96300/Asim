#!/usr/bin/env python3
"""Multi-channel SE verification: tests operator algorithms across channel types.

Compares Python reference inverse vs numpy.linalg.inv across SNR sweep.
Reports SE deviation and pass/fail per channel type.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel, CDLBChannel
from algo import cholesky_noblock_inverse, ldl_noblock_inverse

CHANNELS = {
    "Rayleigh": RayleighChannel(),
    "CDL-B": CDLBChannel(),
    "CDL-B_HighCorr": CDLBChannel(asa_deg=5.0, asd_deg=10.0),  # narrow spread = high corr
}


def compute_se(H, A_inv, noise_power):
    """Spectral efficiency for MMSE detection."""
    B, nr, nt = H.shape
    se = 0.0
    for b in range(B):
        W = A_inv[b] @ H[b].conj().T  # (nt, nr)
        for i in range(nt):
            w = W[i]
            sig = np.abs(w @ H[b, :, i]) ** 2
            interf = sum(np.abs(w @ H[b, :, j]) ** 2 for j in range(nt) if j != i)
            n = noise_power * np.sum(np.abs(w) ** 2)
            se += np.log2(1 + sig / max(interf + n, 1e-15))
    return se / B


def run_se_scan(algo_name, algo_func, batch=96, nr=64, nt=16, snr_db_list=None):
    """Run SE scan across SNR points for a given algorithm."""
    if snr_db_list is None:
        snr_db_list = list(range(0, 31, 5))
    
    results = {}
    for ch_name, ch_gen in CHANNELS.items():
        print(f"\n{'='*50}")
        print(f"  {algo_name} @ {ch_name}")
        print(f"{'='*50}")
        print(f"  {'SNR(dB)':<8} {'SE_algo':<10} {'SE_ref':<10} {'Δ':<10} {'Status'}")
        
        H = ch_gen.generate(batch, nr, nt, seed=42)
        all_pass = True
        
        for snr_db in snr_db_list:
            snr_lin = 10 ** (snr_db / 10.0)
            lam = nt / snr_lin  # MMSE regularization
            noise_power = 1.0 / snr_lin
            
            # Build A = H^H H + lambda*I
            A = np.zeros((batch, nt, nt), dtype=np.complex128)
            for b in range(batch):
                G = H[b].conj().T @ H[b]
                A[b] = G + lam * np.eye(nt)
            
            # Algorithm inverse
            A_inv = np.zeros_like(A)
            for b in range(batch):
                A_inv[b] = algo_func(A[b].copy())
            
            # Reference inverse
            A_ref = np.linalg.inv(A)
            
            # SE comparison
            se_algo = compute_se(H, A_inv, noise_power)
            se_ref = compute_se(H, A_ref, noise_power)
            delta = se_algo - se_ref
            
            status = "PASS" if abs(delta) < 1.0 else "FAIL"
            if abs(delta) >= 1.0:
                all_pass = False
            
            print(f"  {snr_db:<8} {se_algo:<10.4f} {se_ref:<10.4f} {delta:<+10.4f} {status}")
        
        results[ch_name] = {"all_pass": all_pass, "max_delta": max(abs(
            _compute_se_diff(H, algo_func, snr_db_list, nt, batch)
        ) for _ in [1])}
    
    return results


def _compute_se_diff(H, algo_func, snr_db_list, nt, batch):
    """Helper for max delta computation."""
    diffs = []
    for snr_db in snr_db_list:
        snr_lin = 10 ** (snr_db / 10.0)
        lam = nt / snr_lin
        noise_power = 1.0 / snr_lin
        A = np.zeros((batch, nt, nt), dtype=np.complex128)
        for b in range(batch):
            G = H[b].conj().T @ H[b]
            A[b] = G + lam * np.eye(nt)
        A_inv = np.zeros_like(A)
        for b in range(batch):
            A_inv[b] = algo_func(A[b].copy())
        A_ref = np.linalg.inv(A)
        se_algo = compute_se(H, A_inv, noise_power)
        se_ref = compute_se(H, A_ref, noise_power)
        diffs.append(abs(se_algo - se_ref))
    return max(diffs)


if __name__ == "__main__":
    print("=" * 50)
    print("  SE Verification Scan")
    print("  Channels: Rayleigh, CDL-B, CDL-B High-Corr")
    print("  Algorithms: Cholesky NoBlock, LDL NoBlock")
    print("=" * 50)
    
    chol_results = run_se_scan("Cholesky NoBlock", cholesky_noblock_inverse)
    ldl_results = run_se_scan("LDL NoBlock", ldl_noblock_inverse)
    
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    
    all_pass = True
    for ch in CHANNELS:
        chol_ok = chol_results[ch]["all_pass"]
        ldl_ok = ldl_results[ch]["all_pass"]
        print(f"  {ch:<20} Chol={'PASS' if chol_ok else 'FAIL'}  LDL={'PASS' if ldl_ok else 'FAIL'}")
        if not (chol_ok and ldl_ok):
            all_pass = False
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
