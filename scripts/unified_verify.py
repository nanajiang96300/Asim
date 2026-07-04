#!/usr/bin/env python3
"""
Unified verification: same H → Python algo vs C++ DAG replay → direct comparison.
Tests normal (Rayleigh) and harsh (CDL-B High-Corr, low SNR) conditions.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from channel import RayleighChannel, CDLBChannel
from algo import cholesky_noblock_inverse, ldl_noblock_inverse
from uobs_dag_executor import FormulaDAG

CHANNELS = {
    "Rayleigh": RayleighChannel(),
    "CDL-B": CDLBChannel(),
    "CDL-B_Harsh": CDLBChannel(asa_deg=3.0, asd_deg=5.0),  # very narrow spread
}

SNR_LIST = [0, 5, 10, 20, 30]  # dB
BATCH = 96
NR, NT = 64, 16


def fp16c(x):
    r = x.real.astype(np.float16).astype(np.float64)
    i = x.imag.astype(np.float16).astype(np.float64)
    return r + 1j * i


def run_test(algo_name, algo_func, formula_path, seed=42):
    """Run unified test: Python vs DAG on same H matrices."""
    print(f"\n{'='*70}")
    print(f"  {algo_name} — Unified Verification")
    print(f"{'='*70}")
    
    with open(formula_path) as f:
        data = json.load(f)
    dag = FormulaDAG([s for s in data["steps"] if s["batch"] == 0])
    
    results = {}
    for ch_name, ch_gen in CHANNELS.items():
        H = ch_gen.generate(BATCH, NR, NT, seed=seed)
        H_b0 = H[0]  # batch 0 for DAG (single-batch compare)
        
        print(f"\n  --- {ch_name} ---")
        print(f"  {'SNR':<6} {'Py_vs_Ref':<12} {'DAG_vs_Ref':<12} {'Py_vs_DAG':<12} {'SE_ref':<10} {'SE_algo':<10} {'Status'}")
        
        for snr_db in SNR_LIST:
            lam = NT / (10 ** (snr_db / 10.0) + 1e-10)
            
            # Build A from same H
            A_b0 = H_b0.conj().T @ H_b0 + lam * np.eye(NT)
            
            # Reference: numpy.linalg.inv with FP16
            A_ref = fp16c(np.linalg.inv(fp16c(A_b0)))
            
            # Python algorithm
            A_py = algo_func(A_b0.copy())
            A_py = fp16c(A_py)
            
            # C++ DAG replay
            result = dag.execute({"H": H_b0}, {"lambda": lam})
            A_dag = result.get("Ainv")
            A_dag = fp16c(A_dag) if A_dag is not None else None
            
            # Errors
            err_py = np.linalg.norm(A_py - A_ref) / max(np.linalg.norm(A_ref), 1e-15)
            err_dag = np.linalg.norm(A_dag - A_ref) / max(np.linalg.norm(A_ref), 1e-15) if A_dag is not None else float('inf')
            err_cross = np.linalg.norm(A_py - A_dag) / max(np.linalg.norm(A_py), 1e-15) if A_dag is not None else float('inf')
            
            # SE (full batch for Python)
            se_ref = compute_se_batch(H, np.linalg.inv, lam)
            se_py = compute_se_batch(H, algo_func, lam)
            
            status = "PASS" if (err_cross < 0.01 and err_dag < 0.01) else "FAIL"
            print(f"  {snr_db:<6} {err_py:<12.2e} {err_dag:<12.2e} {err_cross:<12.2e} {se_ref:<10.4f} {se_py:<10.4f} {status}")
            
            results[(ch_name, snr_db)] = {
                'err_py': err_py, 'err_dag': err_dag, 'err_cross': err_cross,
                'se_ref': se_ref, 'se_py': se_py,
                'passed': err_cross < 0.01 and err_dag < 0.01
            }
    
    return results


def compute_se_batch(H, inv_func, lam):
    """Compute SE for a batch using either numpy.linalg.inv or algo function."""
    B, nr, nt = H.shape
    A = np.zeros((B, nt, nt), dtype=np.complex128)
    for b in range(B):
        G = H[b].conj().T @ H[b]
        A[b] = G + lam * np.eye(nt)
    
    if inv_func == np.linalg.inv:
        A_inv = np.linalg.inv(A)
    else:
        A_inv = np.zeros_like(A)
        for b in range(B):
            A_inv[b] = inv_func(A[b].copy())
    
    noise_power = lam  # MMSE approximation
    se = 0.0
    for b in range(B):
        W = A_inv[b] @ H[b].conj().T
        for i in range(nt):
            w = W[i]
            sig = np.abs(w @ H[b, :, i]) ** 2
            interf = sum(np.abs(w @ H[b, :, j]) ** 2 for j in range(nt) if j != i)
            n = noise_power * np.sum(np.abs(w) ** 2)
            se += np.log2(1 + sig / max(interf + n, 1e-15))
    return se / B


def run_multi_seed(algo_name, algo_func, formula_path, seeds=(42, 123, 456), batches=(0, 1, 2)):
    """Multi-seed, multi-batch verification."""
    print(f"\n  Multi-Seed + Multi-Batch: {algo_name}")
    with open(formula_path) as f:
        data = json.load(f)

    for ch_name, ch_gen in CHANNELS.items():
        print(f"\n  --- {ch_name} ({len(seeds)} seeds x {len(batches)} batches) ---")
        all_cross = []
        for seed in seeds:
            for batch in batches:
                H = ch_gen.generate(max(batch + 1, BATCH), NR, NT, seed=seed)
                H_b = H[batch]
                dag = FormulaDAG([s for s in data['steps'] if s['batch'] == batch])
                if len(dag.nodes) == 0:
                    print(f"    seed={seed} batch={batch}: no DAG nodes, skipping")
                    continue
                lam = NT / (10 ** (10 / 10.0))  # SNR=10dB
                A_b = H_b.conj().T @ H_b + lam * np.eye(NT)
                A_py = fp16c(algo_func(A_b.copy()))
                try:
                    result = dag.execute({"H": H_b}, {"lambda": lam})
                    A_dag = result.get("Ainv")
                    if A_dag is not None:
                        err = np.linalg.norm(fp16c(A_py) - fp16c(A_dag)) / max(np.linalg.norm(fp16c(A_py)), 1e-15)
                        all_cross.append(err)
                except Exception as e:
                    print(f"    seed={seed} batch={batch}: DAG error: {e}")

        if all_cross:
            mean_err = np.mean(all_cross)
            max_err = np.max(all_cross)
            min_err = np.min(all_cross)
            print(f"    Samples: {len(all_cross)}  Mean: {mean_err:.4e}  Min: {min_err:.4e}  Max: {max_err:.4e}")
            status = "PASS" if max_err < 0.01 else "FAIL"
            print(f"    Status: {status} (threshold=0.01)")
        else:
            print(f"    No valid results — all DAG executions failed")


def compare_intermediates(algo_name, formula_path, ch_name="Rayleigh", seed=42):
    """Compare G, A, L, Y, Ainv between DAG executor and direct Python computation."""
    print(f"\n  Intermediate Tensor Comparison: {algo_name} @ {ch_name}")

    with open(formula_path) as f:
        data = json.load(f)

    ch_gen = CHANNELS[ch_name]
    H = ch_gen.generate(1, NR, NT, seed=seed)[0]

    dag = FormulaDAG([s for s in data['steps'] if s['batch'] == 0])
    lam = NT / (10 ** (10 / 10.0))  # SNR=10dB
    U = NT

    # Direct Python computation of intermediates
    G_py = fp16c(H.conj().T @ H)
    A_py = fp16c(G_py + lam * np.eye(U))

    # DAG execution
    result = dag.execute({"H": H}, {"lambda": lam})

    # Compare known intermediates
    print(f"  {'Tensor':<8} {'DAG shape':<12} {'Error vs Python':<16} {'Status'}")
    all_pass = True

    for name, py_val in [("G", G_py), ("A", A_py)]:
        dag_val = result.get(name)
        if dag_val is not None and py_val is not None:
            err = np.linalg.norm(fp16c(dag_val) - py_val) / max(np.linalg.norm(py_val), 1e-15)
            status = "PASS" if err < 0.01 else "FAIL"
            if err >= 0.01: all_pass = False
            print(f"  {name:<8} {str(dag_val.shape):<12} {err:<16.4e} {status}")

    # For L, Y, Ainv — cross-check with Python algorithm
    A_full = H.conj().T @ H + lam * np.eye(U)
    Ainv_py = fp16c(cholesky_noblock_inverse(A_full.copy()))

    for name in ["L", "Y", "Ainv"]:
        dag_val = result.get(name)
        if dag_val is not None:
            ref = Ainv_py if name == "Ainv" else None
            if ref is not None:
                err = np.linalg.norm(fp16c(dag_val) - ref) / max(np.linalg.norm(ref), 1e-15)
                status = "PASS" if err < 0.01 else "FAIL"
                if err >= 0.01: all_pass = False
                print(f"  {name:<8} {str(dag_val.shape):<12} {err:<16.4e} {status}")
            else:
                print(f"  {name:<8} {str(dag_val.shape):<12} {'(no direct ref)':<16} {'--'}")

    return all_pass


if __name__ == "__main__":
    # Generate formula_steps.json from C++ simulator
    import subprocess
    formula_path = "/tmp/unified_formula.json"
    subprocess.run([
        "./build/bin/Simulator",
        "--config", "configs/ascend_910b_quiet.json",
        "--models_list", "example/cholesky_noblock_v2_test.json",
        "--mode", "cholesky_noblock_v2_test",
        "--log_level", "info"
    ], env={**os.environ, "ONNXIM_FORMULA_JSON": formula_path, "ONNXIM_MAX_CORE_CYCLES": "100000"},
       capture_output=True)
    
    if not os.path.exists(formula_path):
        print("ERROR: formula_steps.json not generated. Run C++ simulator first.")
        sys.exit(1)
    
    print("=" * 70)
    print("  Unified Verification: Python vs C++ DAG Replay")
    print("  Same H → Python algo → A_inv_py")
    print("  Same H → C++ FormulaLogger → DAG executor → A_inv_dag")
    print("=" * 70)
    
    chol_results = run_test("Cholesky NoBlock", cholesky_noblock_inverse, formula_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("  Summary: Python vs C++ DAG Cross-Error")
    print(f"{'='*70}")
    
    all_pass = True
    for ch_name in CHANNELS:
        fails = []
        for snr_db in SNR_LIST:
            r = chol_results[(ch_name, snr_db)]
            if not r['passed']:
                fails.append(snr_db)
        
        if fails:
            print(f"  {ch_name:<20} FAIL at SNR={fails} dB")
            all_pass = False
        else:
            avg_cross = np.mean([chol_results[(ch_name, s)]['err_cross'] for s in SNR_LIST])
            avg_dag = np.mean([chol_results[(ch_name, s)]['err_dag'] for s in SNR_LIST])
            print(f"  {ch_name:<20} PASS  cross_err={avg_cross:.2e}  dag_err={avg_dag:.2e}")
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # Multi-seed + multi-batch verification
    print(f"\n{'='*70}")
    print("  Multi-Seed + Multi-Batch Stability Test")
    print(f"{'='*70}")
    run_multi_seed("Cholesky NoBlock", cholesky_noblock_inverse, formula_path)

    # Intermediate tensor comparison
    print(f"\n{'='*70}")
    print("  Intermediate Tensor Check (G,A,L,Y,Ainv)")
    print(f"{'='*70}")
    for ch in ["Rayleigh", "CDL-B"]:
        compare_intermediates("Cholesky NoBlock", formula_path, ch)
