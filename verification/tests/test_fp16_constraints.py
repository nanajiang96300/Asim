import unittest,json
import numpy as np
from verification.studies.fp16_constraints import diagnose,half
from verification.contracts.core import Invalid

class FP16ConstraintTests(unittest.TestCase):
    def test_oracle_is_explicitly_not_an_fp16_candidate(self):
        rng=np.random.default_rng(53);H=(rng.normal(size=(2,8,4))+1j*rng.normal(size=(2,8,4)))/4;Y=rng.normal(size=(2,8,2))+1j*rng.normal(size=(2,8,2))
        r=diagnose(H,Y,.01);json.dumps(r,allow_nan=False)
        self.assertTrue(r['reference_only']);self.assertFalse(r['candidate_verified'])
        self.assertLess(r['output_inverse_fp16_storage_error_max'],.001)
    def test_no_overflow_or_invalid_input_is_repaired(self):
        with self.assertRaises((Invalid,FloatingPointError)):half(np.array([1e10]))
        with self.assertRaises(Invalid):diagnose(np.zeros((2,8,4)),np.zeros((2,7,2)),.01)
