"""Bounded FP16 algorithm screening before candidate implementation.

This module deliberately does not import Asim/reconstruction or issue scores.
It reuses independent references, not a second implementation of the algorithm.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from verification.contracts.core import Invalid, Unsupported, digest
from verification.contracts.mmse import MMSETask, fixture, task_reference
from verification.operators.parameters import ns_parameters, bri_parameters
from verification.reference.ns import simulate_ns_fp16_model
from verification.reference.bri import simulate_bri_fp16_model

PROFILE = 'modeled_fp16_fp32_real_block_v1'
ALGORITHMS = {
    'ns_corrected_v1': (32, simulate_ns_fp16_model, ns_parameters),
    'bri_explicit_v1': (256, simulate_bri_fp16_model, bri_parameters),
}


def validate(spec):
    required = {'schema', 'operator', 'task', 'regularizations', 'calibration_seeds',
                'holdout_seeds', 'iterations', 'max_task_error', 'max_reference_calls',
                'required_metrics'}
    if set(spec) != required or spec['schema'] != 'iteration-screen-v1':
        raise Invalid('unknown/incomplete iteration contract')
    if spec['operator'] not in ALGORITHMS:
        raise Unsupported('no independent FP16 reference for operator')
    if spec['required_metrics'] != ['mmse_output_relative_error']:
        raise Unsupported('SE/BER gates are not implemented by this offline screen')
    task = MMSETask(**spec['task']); task.validate()
    if task.profile != PROFILE or task.scope != 'mmse_h_y_to_xhat_v1':
        raise Unsupported('screen requires full-task explicit FP16 model')
    groups = [spec['calibration_seeds'], spec['holdout_seeds']]
    for seeds in groups:
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
            raise Invalid('distinct nonnegative integer seeds required')
    if set(groups[0]) & set(groups[1]):
        raise Invalid('calibration and holdout seeds overlap')
    values = spec['regularizations']
    if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values) or len(set(values)) != len(values):
        raise Invalid('distinct positive finite regularizations required')
    if task.regularization != values[0]:
        raise Invalid('task regularization must identify first scenario')
    ks = spec['iterations']
    if not isinstance(ks, list) or not ks or any(type(k) is not int or not 1 <= k <= ALGORITHMS[spec['operator']][0] for k in ks) or ks != sorted(set(ks)):
        raise Invalid('ordered distinct supported iteration counts required')
    limit = spec['max_task_error']
    if type(limit) not in (float, int) or not math.isfinite(limit) or limit <= 0:
        raise Invalid('finite positive error threshold required')
    budget = spec['max_reference_calls']
    calls = len(values) * (len(ks) * len(groups[0]) + len(groups[1]))
    if type(budget) is not int or budget <= 0 or calls > budget:
        raise Invalid('reference call budget exceeded')
    return task


def assess(output, reference):
    output = np.asarray(output); reference = np.asarray(reference)
    if output.shape != reference.shape or not np.isfinite(output).all() or not np.isfinite(reference).all():
        raise Invalid('reference model output shape/nonfinite failure')
    errors = []
    for x, truth in zip(output, reference):
        scale = max(1., float(np.max(np.abs(x))), float(np.max(np.abs(truth))))
        errors.append(float(np.linalg.norm((x / scale) - (truth / scale)) /
                            max(np.linalg.norm(truth / scale), 1e-15 / scale)))
    if not errors or not all(math.isfinite(e) for e in errors):
        raise Invalid('invalid derived task errors')
    return errors


def screen(spec, case_provider=None, simulate=None):
    task = validate(spec)
    model = simulate or ALGORITHMS[spec['operator']][1]
    if case_provider is None:
        def case_provider(t, seed):
            data = fixture(t, seed)
            return data, task_reference(data, t)
    records = []
    identities = []

    def evaluate(k, seeds, split):
        passed = True
        for lam in spec['regularizations']:
            t = MMSETask(**{**asdict(task), 'regularization': lam})
            for seed in seeds:
                data, truth = case_provider(t, seed)
                # Hash original and quantized inputs: reference and modeled paths differ.
                arrays = [data['H'], data['Y'], truth, *[data['inputs'][n] for n in sorted(data['inputs'])]]
                identity = digest({'task': t.frozen(), 'seed': seed, 'arrays': [
                    {'shape': list(a.shape), 'dtype': str(a.dtype),
                     'sha256': hashlib.sha256(a.tobytes()).hexdigest()} for a in arrays]})
                identities.append({'split': split, 'seed': seed, 'regularization': lam, 'identity': identity})
                try:
                    result = model(data['H'], data['Y'], lam, k)
                    errors = assess(result['output'], truth)
                    ok = max(errors) <= spec['max_task_error']
                    record = {'errors_per_batch': errors, 'worst_error': max(errors), 'passed': ok}
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    record = {'passed': False, 'failure': type(exc).__name__ + ': ' + str(exc)}
                    ok = False
                records.append({'split': split, 'iterations': k, 'seed': seed, 'regularization': lam, **record})
                passed = passed and ok
        return passed

    # Enumerate: FP16 error is not assumed monotonic. Exhaust the frozen grid.
    passing = [k for k in spec['iterations'] if evaluate(k, spec['calibration_seeds'], 'calibration')]
    selected = min(passing) if passing else None
    holdout_passed = evaluate(selected, spec['holdout_seeds'], 'holdout') if selected is not None else False
    return {
        'schema': 'iteration-screen-result-v1', 'contract': spec, 'contract_sha256': digest(spec),
        'status': ('ALGORITHM_SCREENED' if holdout_passed else
                   'HOLDOUT_FAILED' if selected is not None else 'NO_FEASIBLE_K'),
        'selected_iterations': selected, 'calibration_passing_iterations': passing,
        'minimality': 'all_integers_up_to_selected' if selected is not None and
                      all(k in spec['iterations'] for k in range(1, selected + 1)) else 'frozen_grid_only',
        'algorithm_parameters': ALGORITHMS[spec['operator']][2](selected) if selected is not None else None,
        'cases': records, 'input_identities': identities,
        'operator_verified': False, 'communication_status': 'NOT_RUN', 'score': None,
        'next_gate': 'FREEZE_IMPLEMENTATION_THEN_ASIM_SCREEN_THEN_RECONSTRUCTION',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise Invalid('refusing existing archive directory')
    spec = json.loads(args.contract.read_text()); validate(spec)
    start = time.perf_counter()
    result = screen(spec)
    repo = Path(__file__).resolve().parents[2]
    # Preserve the exact implementation dependency closure used by the screen.
    paths = [__file__, 'verification/planning/__init__.py', 'verification/contracts/core.py',
             'verification/contracts/mmse.py', 'verification/fixtures/rayleigh.py',
             'verification/profiles/numeric.py', 'verification/operators/parameters.py',
             'verification/reference/ns.py', 'verification/reference/bri.py']
    sources = {}
    for name in paths:
        p = Path(name) if Path(name).is_absolute() else repo / name
        sources[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result.update(provenance={'sources': sources, 'python': platform.python_version(), 'numpy': np.__version__},
                  elapsed_seconds=time.perf_counter() - start, pure_model_thinking_seconds=None)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'contract.json').write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'selected_iterations', 'communication_status', 'elapsed_seconds')}))
    return 0 if result['status'] == 'ALGORITHM_SCREENED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
