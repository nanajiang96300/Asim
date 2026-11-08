"""Execute an approved quality plan serially through the pinned Asim-first core."""
import argparse
import json
import os
import time
from pathlib import Path

from verification.contracts.core import Invalid, ScreenPolicy, digest
from verification.contracts.mmse import MMSETask
from verification.planning.handoff import verify
from verification.profiles.numeric import policy_for
from verification.reconstruction.daily import sha
from verification.reconstruction.runner import evaluate

PASSED = 'MODELED_NUMERIC_CHECKED'


def approved(path, expected_sha, repo):
    if sha(path) != expected_sha:
        raise Invalid('approved plan SHA-256 mismatch')
    plan = json.loads(Path(path).read_text())
    return plan, verify(plan, repo)


def seal_job(directory, result, plan, job, runtime):
    """Preserve the core receipt, then attach the stricter planning gate."""
    raw = directory / 'result.json'
    if json.loads(raw.read_text()) != result:
        raise Invalid('core result changed before sealing')
    names = set()
    for artifact in result['artifacts']:
        name = artifact['path']; p = directory / name
        if name in names or Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file() or p.is_symlink() or p.stat().st_size != artifact['bytes'] or sha(p) != artifact['sha256']:
            raise Invalid('core artifact changed/unsafe before sealing')
        names.add(name)
    (directory / 'evaluation_result.json').write_bytes(raw.read_bytes())
    (directory / 'runtime_pin.json').write_text(json.dumps(dict(
        spec_sha256=plan['runtime_sha256'], files=runtime['files'],
        runtime_contract=runtime['runtime_contract'], binary_source_commit=runtime['binary_source_commit']), indent=2))
    result['runtime_status'] = result['status']
    result['plan_quality_status'] = 'NOT_RUN'
    if result['numeric_status'] == 'EXECUTED':
        rows = result.get('batches', [])
        limit = plan['screen']['contract']['max_task_error']
        good = (result['status'] == PASSED and len(rows) == job['task']['batch_size'] and
                {row['batch'] for row in rows} == set(range(job['task']['batch_size'])) and
                all(row.get('precision_bit_equal') is True and row.get('task_error', float('inf')) <= limit for row in rows))
        result['plan_quality_status'] = 'PASSED' if good else 'FAILED'
        if result['status'] == PASSED and not good: result['status'] = 'PLAN_QUALITY_FAILED'
    if result.get('asim') and result['asim']['binary_sha256'] != sha(Path(__file__).resolve().parents[2] / runtime['binary']):
        raise Invalid('evaluated binary differs from approved runtime')
    result.update(entrypoint='supervised_quality_plan_v1', communication_status='NOT_RUN', score=None)
    (directory / 'plan_binding.json').write_text(json.dumps(dict(
        plan_content_sha256=digest(plan),
        contract_sha256=plan['screen']['contract_sha256'], algorithm=plan['algorithm'],
        job=job, plan_quality_status=result['plan_quality_status']), indent=2))
    for name in ('evaluation_result.json', 'runtime_pin.json', 'plan_binding.json'):
        p = directory / name
        result['artifacts'].append(dict(path=name, bytes=p.stat().st_size, sha256=sha(p)))
    raw.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def run(plan_path, expected_sha, output, repo):
    output = Path(output).resolve(); repo = Path(repo).resolve()
    if output.exists(): raise Invalid('refusing existing execution directory')
    start = time.perf_counter(); phase = 'PREFLIGHT'; rows = []
    result = dict(schema='quality-plan-execution-v1', status='PREFLIGHT_REJECTED',
                  numeric_status='NOT_RUN', communication_status='NOT_RUN', score=None,
                  approved_plan_sha256=expected_sha)
    try:
        plan, runtime = approved(plan_path, expected_sha, repo)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'approved_plan.json').write_bytes(Path(plan_path).read_bytes())
        if runtime.get('library_dir'): os.environ['LD_LIBRARY_PATH'] = str(repo / runtime['library_dir'])
        os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['OMP_NUM_THREADS'] = '1'
        phase = 'EXECUTION'; failed = False
        for job in plan['jobs']:
            if failed:
                rows.append(dict(id=job['id'], status='NOT_RUN', numeric_status='NOT_RUN'))
                continue
            approved(plan_path, expected_sha, repo)
            task = MMSETask(**job['task']); config = json.loads((repo / runtime['config']).read_text())
            config['precision'] = policy_for(task.profile).bytes
            for key in ('dram_config_path', 'icnt_config_path'):
                if config.get(key): config[key] = str((repo / runtime['cwd'] / config[key]).resolve())
            b = job.get('baseline'); policy = ScreenPolicy(**b['policy']) if b else None
            directory = output / job['id']
            actual = evaluate(plan['operator'], task, plan['algorithm']['iterations'], job['seed'],
                              repo / runtime['binary'], config, directory, directory,
                              policy, b['comparison_identity'] if b else None)
            if actual.get('numeric_status') == 'EXECUTED': result['numeric_status'] = 'EXECUTED'
            approved(plan_path, expected_sha, repo)
            if actual.get('task') is not None and (actual['task'] != task.frozen() or actual['algorithm'] != plan['algorithm'] or actual['seed'] != job['seed']):
                raise Invalid('actual task/algorithm/input seed differs from approved job')
            actual = seal_job(directory, actual, plan, job, runtime)
            rows.append(dict(id=job['id'], split=job['split'], seed=job['seed'],
                             regularization=task.regularization, status=actual['status'],
                             numeric_status=actual['numeric_status'], plan_quality_status=actual['plan_quality_status'],
                             finish_cycle=actual.get('asim', {}).get('finish_cycle'),
                             result_sha256=sha(directory / 'result.json')))
            print(json.dumps(rows[-1]), flush=True)
            failed = actual['status'] != PASSED or actual['plan_quality_status'] != 'PASSED'
        approved(plan_path, expected_sha, repo)
        result.update(status='PLAN_GATE_FAILED' if failed else 'PLAN_RECONSTRUCTION_CHECKED',
                      numeric_status='EXECUTED' if any(r['numeric_status'] == 'EXECUTED' for r in rows) else 'NOT_RUN',
                      baseline_creation=plan['execution']['baseline_creation'])
    except (Invalid, ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result.update(status='PREFLIGHT_REJECTED' if phase == 'PREFLIGHT' else 'RUN_OR_ARCHIVE_FAILED', reason=str(exc), phase=phase)
    output.mkdir(parents=True, exist_ok=True)
    result.update(jobs=rows, elapsed_seconds=time.perf_counter() - start, pure_model_thinking_seconds=None)
    result['artifacts'] = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha(p))
                           for p in sorted(output.rglob('*')) if p.is_file()]
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    result = run(a.plan, a.plan_sha256, a.output, Path(__file__).resolve().parents[2])
    print(json.dumps({k: result[k] for k in ('status', 'numeric_status', 'communication_status', 'elapsed_seconds')}))
    return 0 if result['status'] == 'PLAN_RECONSTRUCTION_CHECKED' else 1


if __name__ == '__main__': raise SystemExit(main())
