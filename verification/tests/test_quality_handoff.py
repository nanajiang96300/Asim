import copy
import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask
from verification.planning.handoff import freeze, verify, fingerprints, SCREEN_SOURCES
from verification.planning.iterations import screen
from verification.planning.execution import approved, run, seal_job
from verification.reconstruction.daily import sha
from verification.schedule.runtime import HEADER
from verification.tests.test_iteration_planning import contract

REPO = Path(__file__).resolve().parents[2]


class QualityHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = contract(); spec['iterations'] = [1, 12]; spec['max_reference_calls'] = 3
        cls.screened = screen(spec)
        cls.screened.update(provenance=dict(sources=fingerprints(REPO, SCREEN_SOURCES),
                                            python=platform.python_version(), numpy=np.__version__),
                            elapsed_seconds=0., pure_model_thinking_seconds=None)

    def fixture(self, directory):
        root = Path(directory)
        for name in ('binary', 'config'):
            (root / name).write_text('{}')
        spec = dict(schema='supervised-reconstruction-runtime-v1', runtime_contract=HEADER,
                    binary=str(root / 'binary'), config=str(root / 'config'), dependencies=[],
                    cwd=str(root), binary_source_commit='unit-test',
                    operators=['ns_corrected_v1'], profiles=['modeled_fp16_fp32_real_block_v1'],
                    python_version=platform.python_version(), numpy_version=np.__version__,
                    files=[dict(path=str(root / n), bytes=2, sha256=sha(root / n)) for n in ('binary', 'config')])
        runtime = root / 'runtime.json'; runtime.write_text(json.dumps(spec))
        screened = root / 'screen.json'; screened.write_text(json.dumps(self.screened))
        plan = freeze(screened, runtime, REPO, create_baseline=True)
        path = root / 'plan.json'; path.write_text(json.dumps(plan))
        return path, plan

    def test_approved_plan_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            p, plan = self.fixture(d)
            approved(p, sha(p), REPO)
            self.assertEqual([j['split'] for j in plan['jobs']], ['calibration', 'holdout'])
            self.assertEqual(plan['algorithm']['iterations'], 12)

    def test_approved_hash_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.fixture(d); expected = sha(p); p.write_text(p.read_text() + ' ')
            with self.assertRaises(Invalid): approved(p, expected, REPO)

    def test_task_iteration_precision_seed_and_coverage_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            _, plan = self.fixture(d)
            for field, value in [('seed', 999), ('split', 'holdout'), ('task', dict(plan['jobs'][0]['task'], profile='structure_f64_real_block'))]:
                bad = copy.deepcopy(plan); bad['jobs'][0][field] = value
                with self.subTest(field=field), self.assertRaises(Invalid): verify(bad, REPO)
            for change in ('iterations', 'coverage', 'sources', 'promotion'):
                bad = copy.deepcopy(plan)
                if change == 'iterations': bad['algorithm']['iterations'] = 1
                elif change == 'coverage': bad['jobs'].pop()
                elif change == 'sources': bad['binding_sources'].pop('orchestrator/codex_operators.json')
                else: bad['execution']['score'] = 100
                with self.subTest(change=change), self.assertRaises(Invalid): verify(bad, REPO)

    def test_changed_input_source_and_binary_fail(self):
        for target in ('screen', 'binary'):
            with tempfile.TemporaryDirectory() as d:
                _, plan = self.fixture(d)
                Path(plan['screen_source'] if target == 'screen' else Path(d) / 'binary').write_text('changed')
                with self.subTest(target=target), self.assertRaises(Invalid): verify(plan, REPO)

    def test_freeze_recomputes_claimed_screen(self):
        with tempfile.TemporaryDirectory() as d:
            _, plan = self.fixture(d)
            p = Path(plan['screen_source']); bad = json.loads(p.read_text())
            bad['cases'][0]['errors_per_batch'][0] = 0
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(Invalid, 'recomputation'):
                freeze(p, Path(d) / 'runtime.json', REPO, create_baseline=True)

    def test_baseline_creation_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            _, plan = self.fixture(d)
            with self.assertRaises(Invalid): freeze(plan['screen_source'], Path(d) / 'runtime.json', REPO)
            with self.assertRaises(Invalid):
                freeze(plan['screen_source'], Path(d) / 'runtime.json', REPO, create_baseline=True, minimum_speedup=2)

    def test_preflight_failure_never_calls_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            p, _ = self.fixture(d)
            with patch('verification.planning.execution.evaluate') as evaluator:
                r = run(p, 'incorrect', Path(d) / 'output', REPO)
                evaluator.assert_not_called()
            self.assertEqual(r['status'], 'PREFLIGHT_REJECTED')
            self.assertEqual(r['numeric_status'], 'NOT_RUN')

    def test_runtime_change_after_evaluation_revokes_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            p, plan = self.fixture(d)
            def changed(*args, **kwargs):
                Path(plan['runtime_path']).write_text(Path(plan['runtime_path']).read_text() + ' ')
                return dict(status='MODELED_NUMERIC_CHECKED', numeric_status='EXECUTED')
            with patch('verification.planning.execution.evaluate', side_effect=changed):
                r = run(p, sha(p), Path(d) / 'output', REPO)
            self.assertEqual(r['status'], 'RUN_OR_ARCHIVE_FAILED')
            self.assertEqual(r['numeric_status'], 'EXECUTED')
            self.assertIn('manifest changed', r['reason'])

    def test_source_change_after_freeze_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, plan = self.fixture(d)
            changed = dict(plan['binding_sources']); changed['verification/planning/execution.py'] = 'new'
            with patch('verification.planning.handoff.fingerprints', return_value=changed):
                with self.assertRaisesRegex(Invalid, 'source/registry'): verify(plan, REPO)

    def test_asim_failure_skips_all_numeric_functions(self):
        from verification.reconstruction.runner import evaluate
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / 'run'
            def failed(*args, **kwargs):
                directory.mkdir()
                return dict(status='FAILED', returncode=1, finish_cycle=None, artifacts=[])
            config = dict(num_cores=24, precision=2, dram_req_size=64,
                          ascend_cube_model=dict(enabled=True, cube_m=16, cube_n=16, cube_k=16))
            with patch('verification.reconstruction.runner.simulate', side_effect=failed), \
                 patch('verification.reference.ns.simulate_ns_fp16_model') as reference, \
                 patch('verification.reconstruction.runner.execute_batch') as replay:
                r = evaluate('ns_corrected_v1', MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1'),
                             12,42,'unused',config,directory,directory)
                reference.assert_not_called(); replay.assert_not_called()
            self.assertEqual(r['status'], 'SKIPPED_ASIM_FAILED')
            self.assertEqual(r['numeric_status'], 'NOT_RUN')
            self.assertFalse((directory / 'numerics_started.json').exists())

    def test_stricter_plan_threshold_cannot_be_lost_at_handoff(self):
        with tempfile.TemporaryDirectory() as d:
            p, plan = self.fixture(d); plan['screen']['contract']['max_task_error'] = .001
            root = Path(d) / 'job'; root.mkdir()
            r = dict(status='MODELED_NUMERIC_CHECKED', numeric_status='EXECUTED', artifacts=[],
                     batches=[dict(batch=b,precision_bit_equal=True,task_error=.01) for b in range(24)])
            (root / 'result.json').write_text(json.dumps(r))
            runtime = verify(json.loads(p.read_text()), REPO)
            sealed = seal_job(root, r, plan, plan['jobs'][0], runtime)
            self.assertEqual(sealed['status'], 'PLAN_QUALITY_FAILED')
            self.assertIsNone(sealed['score'])

    def test_seal_compares_json_semantics_without_ignoring_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p, plan = self.fixture(d); runtime = verify(plan, REPO)
            root = Path(d) / 'job'; root.mkdir()
            r = dict(status='SKIPPED_ASIM_FAILED', numeric_status='NOT_RUN', artifacts=[], core_tile_counts={0: (1, 2)})
            (root / 'result.json').write_text(json.dumps(r))
            sealed = seal_job(root, r, plan, plan['jobs'][0], runtime)
            self.assertEqual(sealed['core_tile_counts'], {'0': [1, 2]})
            # Recreate original core record, then change an actual count.
            (root / 'result.json').write_text(json.dumps(r)); r['core_tile_counts'][0] = (1, 3)
            with self.assertRaisesRegex(Invalid, 'changed before sealing'):
                seal_job(root, r, plan, plan['jobs'][0], runtime)


if __name__ == '__main__': unittest.main()
