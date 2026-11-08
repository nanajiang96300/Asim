import copy
import unittest

import numpy as np

from verification.contracts.core import Invalid, Unsupported
from verification.contracts.mmse import fixture, task_reference
from verification.planning.iterations import screen, validate, assess
from verification.reference.ns import simulate_ns_fp16_model


def contract():
    return dict(schema='iteration-screen-v1', operator='ns_corrected_v1',
                task=dict(batch_size=24, nr=8, nt=4, nrhs=2, regularization=.1,
                          profile='modeled_fp16_fp32_real_block_v1'),
                regularizations=[.1], calibration_seeds=[42], holdout_seeds=[123],
                iterations=[1, 2, 3], max_task_error=.02, max_reference_calls=4,
                required_metrics=['mmse_output_relative_error'])


def case(t, seed):
    a = np.full((t.batch_size, 1, 1), seed, dtype=float)
    return dict(H=a, Y=a, inputs={'H': a}), np.ones_like(a)


class IterationPlanningTests(unittest.TestCase):
    def test_nonmonotonic_grid_and_minimum(self):
        def model(h, y, lam, k):
            return {'output': np.full_like(h, {1: 1.4, 2: 1.001, 3: 1.3}[k])}
        r = screen(contract(), case, model)
        self.assertEqual(r['selected_iterations'], 2)
        self.assertEqual(r['calibration_passing_iterations'], [2])
        self.assertEqual(r['status'], 'ALGORITHM_SCREENED')
        self.assertFalse(r['operator_verified'])
        self.assertIsNone(r['score'])

    def test_holdout_cannot_retune(self):
        calls = []
        def model(h, y, lam, k):
            calls.append((int(h[0, 0, 0]), k))
            return {'output': np.ones_like(h) * (1 if h[0, 0, 0] == 42 or k == 3 else 2)}
        r = screen(contract(), case, model)
        self.assertEqual(r['selected_iterations'], 1)
        self.assertEqual(r['status'], 'HOLDOUT_FAILED')
        self.assertEqual([k for seed, k in calls if seed == 123], [1])

    def test_no_feasible_and_nonfinite(self):
        for value in (3., np.nan):
            r = screen(contract(), case, lambda h, y, lam, k: {'output': np.full_like(h, value)})
            self.assertEqual(r['status'], 'NO_FEASIBLE_K')
            self.assertIsNone(r['selected_iterations'])
            self.assertFalse(any(c['split'] == 'holdout' for c in r['cases']))

    def test_worst_batch_not_average(self):
        def model(h, y, lam, k):
            x = np.ones_like(h); x[0] = 1.1
            return {'output': x}
        self.assertEqual(screen(contract(), case, model)['status'], 'NO_FEASIBLE_K')

    def test_sparse_grid_does_not_claim_global_minimum(self):
        s = contract(); s['iterations'] = [2, 3]
        r = screen(s, case, lambda h, y, lam, k: {'output': np.ones_like(h)})
        self.assertEqual(r['minimality'], 'frozen_grid_only')

    def test_fail_closed_contract(self):
        changes = [('holdout_seeds', [42]), ('max_reference_calls', 3),
                   ('iterations', [True]), ('iterations', [3, 1]),
                   ('regularizations', [.1, .1]), ('max_task_error', float('nan')),
                   ('required_metrics', ['se_loss']), ('extra', True)]
        for key, value in changes:
            s = copy.deepcopy(contract()); s[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                validate(s)
        s = contract(); s['task']['channel'] = 'cdl-b'
        with self.assertRaises(Unsupported): validate(s)

    def test_matching_reference_uses_all_input_quantization(self):
        s = contract(); s['iterations'] = [1, 12]; s['max_reference_calls'] = 3
        r = screen(s)
        t = validate(s); d = fixture(t, 42)
        expected = assess(simulate_ns_fp16_model(d['H'], d['Y'], .1, 12)['output'], task_reference(d, t))
        actual = next(c for c in r['cases'] if c['split'] == 'calibration' and c['iterations'] == 12)
        np.testing.assert_array_equal(actual['errors_per_batch'], expected)
        self.assertEqual(r['selected_iterations'], 12)

    def test_all_regularization_scenarios_must_pass(self):
        s = contract(); s['regularizations'] = [.1, 1.]; s['max_reference_calls'] = 8
        def model(h, y, lam, k):
            return {'output': np.ones_like(h) * (1 if lam == .1 else 2)}
        self.assertEqual(screen(s, case, model)['status'], 'NO_FEASIBLE_K')


if __name__ == '__main__': unittest.main()
