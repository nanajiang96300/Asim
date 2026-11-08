"""Trusted, supervised handoff from independently recomputed screen to runtime.

SHA-256 binds a supervisor-approved plan, not a digital signature or OS sandbox.
"""
import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from verification.contracts.core import Invalid, ScreenPolicy, digest
from verification.contracts.mmse import MMSETask
from verification.planning.iterations import ALGORITHMS, screen, validate
from verification.profiles.numeric import policy_for
from verification.operators.catalog import adapter
from verification.reconstruction.daily import load_runtime, baseline, sha

SCREEN_SOURCES = [
    'verification/planning/iterations.py', 'verification/planning/__init__.py',
    'verification/contracts/core.py', 'verification/contracts/mmse.py',
    'verification/fixtures/rayleigh.py', 'verification/profiles/numeric.py',
    'verification/operators/parameters.py', 'verification/reference/ns.py',
    'verification/reference/bri.py',
]
BINDING_SOURCES = SCREEN_SOURCES + [
    'verification/planning/handoff.py', 'verification/planning/execution.py',
    'verification/operators/catalog.py', 'orchestrator/codex_operators.json',
]


def fingerprints(repo, paths):
    return {p: sha(repo / p) for p in paths}


def jobs_for(result):
    spec = result['contract']; task = validate(spec)
    # Both splits are replayed. These seeds no longer constitute a fresh test set.
    jobs = []
    for split in ('calibration', 'holdout'):
        for lam in spec['regularizations']:
            for seed in spec[split + '_seeds']:
                task_dict = {**asdict(task), 'regularization': lam}
                jobs.append(dict(id=f'case{len(jobs):03d}', split=split, seed=seed, task=task_dict))
    if len(jobs) > 8:
        raise Invalid('supervised handoff supports at most 8 serial runs per plan')
    return jobs


def validate_screen(result, repo, recompute=False):
    task = validate(result['contract'])
    provenance = result['provenance']
    if provenance != dict(sources=fingerprints(repo, SCREEN_SOURCES),
                          python=platform.python_version(), numpy=np.__version__):
        raise Invalid('screen source/toolchain provenance changed or incomplete')
    if result['contract_sha256'] != digest(result['contract']) or result['status'] != 'ALGORITHM_SCREENED':
        raise Invalid('only intact successful algorithm screens may be frozen')
    k = result['selected_iterations']
    if type(k) is not int or k not in result['contract']['iterations']:
        raise Invalid('invalid selected iteration count')
    if result['contract']['max_task_error'] > policy_for(task.profile).task_tolerance:
        raise Invalid('plan cannot relax the accepted runtime numeric threshold')
    _, params, _, reference = adapter(result['contract']['operator'], task, k)
    if params != dict(id=result['contract']['operator'], **result['algorithm_parameters']) or reference is not ALGORITHMS[result['contract']['operator']][1]:
        raise Invalid('screen and actual runtime adapter/reference disagree')
    if recompute:
        expected = screen(result['contract'])
        extra = {'provenance', 'elapsed_seconds', 'pure_model_thinking_seconds'}
        if {key: value for key, value in result.items() if key not in extra} != expected:
            raise Invalid('screen claims differ from independent recomputation')
    return params


def freeze(screen_path, runtime_path, repo, create_baseline=False, baselines=(),
           minimum_speedup=1., maximum_speedup=10.):
    repo = Path(repo).resolve(); screen_path = Path(screen_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    sources = fingerprints(repo, BINDING_SOURCES)
    original = sha(screen_path); result = json.loads(screen_path.read_text())
    params = validate_screen(result, repo, recompute=True)
    runtime = load_runtime(runtime_path, repo)
    if result['provenance']['python'] != runtime['python_version'] or result['provenance']['numpy'] != runtime['numpy_version']:
        raise Invalid('screen/runtime toolchain mismatch')
    spec = result['contract']; op = spec['operator']
    if op not in runtime['operators'] or spec['task'].get('profile') not in runtime['profiles']:
        raise Invalid('operator/profile not accepted by pinned runtime')
    jobs = jobs_for(result)
    if create_baseline:
        if baselines or minimum_speedup != 1. or maximum_speedup != 10.:
            raise Invalid('baseline creation cannot also claim a performance improvement')
    elif len(baselines) != len(jobs):
        raise Invalid('provide one sealed baseline per job, or explicitly create baselines')
    for job, path in zip(jobs, baselines):
        path = Path(path).resolve(); blob = sha(path)
        cycles, identity = baseline(path, op, result['selected_iterations'], sha(repo / runtime['binary']), runtime['_loaded_sha256'], params)
        previous = json.loads(path.read_text())
        if previous['task'] != MMSETask(**job['task']).frozen() or previous['seed'] != job['seed']:
            raise Invalid('baseline task/seed differs from plan job')
        policy = ScreenPolicy(cycles, minimum_speedup, maximum_speedup); policy.validate()
        job['baseline'] = dict(path=str(path), sha256=blob, comparison_identity=identity, policy=vars(policy))
    if sha(screen_path) != original or fingerprints(repo, BINDING_SOURCES) != sources or sha(runtime_path) != runtime['_loaded_sha256']:
        raise Invalid('screen/runtime/sources changed while freezing')
    return dict(schema='quality-execution-plan-v1', operator=op,
                screen=result, screen_sha256=original, screen_source=str(screen_path),
                runtime_path=str(runtime_path), runtime_sha256=runtime['_loaded_sha256'],
                binding_sources=sources, algorithm=params, jobs=jobs,
                execution=dict(serial=True, max_jobs=8, baseline_creation=create_baseline,
                               communication_status='NOT_RUN', score=None))


def verify(plan, repo):
    """Check the complete approved plan, without rerunning numerical screening."""
    required = {'schema', 'operator', 'screen', 'screen_sha256', 'screen_source',
                'runtime_path', 'runtime_sha256', 'binding_sources', 'algorithm', 'jobs', 'execution'}
    if set(plan) != required or plan['schema'] != 'quality-execution-plan-v1':
        raise Invalid('invalid execution plan schema')
    if plan['binding_sources'] != fingerprints(repo, BINDING_SOURCES):
        raise Invalid('plan binding source/registry changed')
    if sha(plan['screen_source']) != plan['screen_sha256'] or json.loads(Path(plan['screen_source']).read_text()) != plan['screen']:
        raise Invalid('frozen screen evidence changed')
    params = validate_screen(plan['screen'], repo)
    if plan['algorithm'] != params or plan['operator'] != plan['screen']['contract']['operator']:
        raise Invalid('algorithm/iteration override rejected')
    jobs = jobs_for(plan['screen'])
    if len(jobs) != len(plan['jobs']):
        raise Invalid('plan job coverage mismatch')
    creation = plan['execution']['baseline_creation']
    if type(creation) is not bool or plan['execution'] != dict(serial=True, max_jobs=8, baseline_creation=creation, communication_status='NOT_RUN', score=None):
        raise Invalid('unrecognized execution or promotion policy')
    runtime = load_runtime(plan['runtime_path'], repo)
    if runtime['_loaded_sha256'] != plan['runtime_sha256']:
        raise Invalid('approved runtime manifest changed')
    if plan['operator'] not in runtime['operators'] or plan['screen']['contract']['task']['profile'] not in runtime['profiles']:
        raise Invalid('runtime no longer accepts plan')
    if runtime['python_version'] != platform.python_version() or runtime['numpy_version'] != np.__version__:
        raise Invalid('runtime toolchain mismatch')
    for actual, expected in zip(plan['jobs'], jobs):
        b = actual.get('baseline')
        if {k: v for k, v in actual.items() if k != 'baseline'} != expected or (b is not None) == creation:
            raise Invalid('task/input/seed/scope/coverage override rejected')
        if b is not None:
            if sha(b['path']) != b['sha256']:
                raise Invalid('baseline result changed')
            cycles, identity = baseline(b['path'], plan['operator'], params['iterations'],
                                        sha(repo / runtime['binary']), plan['runtime_sha256'], params)
            p = ScreenPolicy(**b['policy']); p.validate()
            previous = json.loads(Path(b['path']).read_text())
            if cycles != p.baseline_cycles or identity != b['comparison_identity'] or previous['seed'] != actual['seed'] or previous['task'] != MMSETask(**actual['task']).frozen():
                raise Invalid('baseline no longer matches plan job')
    return runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--screen', required=True, type=Path)
    ap.add_argument('--runtime', type=Path, default=Path('orchestrator/codex_runtime.json'))
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--create-baseline', action='store_true')
    ap.add_argument('--baseline', type=Path, action='append', default=[])
    ap.add_argument('--minimum-speedup', type=float, default=1.)
    ap.add_argument('--maximum-speedup', type=float, default=10.)
    a = ap.parse_args()
    if a.output.exists(): raise Invalid('refusing existing plan path')
    plan = freeze(a.screen, a.runtime, Path(__file__).resolve().parents[2], a.create_baseline,
                  a.baseline, a.minimum_speedup, a.maximum_speedup)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: f.write(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(plan=str(a.output), plan_sha256=sha(a.output), jobs=len(plan['jobs']))))


if __name__ == '__main__': raise SystemExit(main())
