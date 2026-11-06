import unittest
import numpy as np
from verification.operators.catalog import adapter
from verification.contracts.mmse import MMSETask,fixture,task_reference
from verification.contracts.core import Invalid
from verification.reference.cholesky import simulate_cholesky,simulate_cholesky_fp16_model
from verification.reporting.effort import summarize

class GeneralityTests(unittest.TestCase):
    def test_direct_adapter_has_no_ns_initialization(self):
        _,p,_,_=adapter('cholesky_explicit_v1',MMSETask(),1)
        self.assertNotIn('iterations',p);self.assertNotIn('trace',p['initialization'])
        with self.assertRaises(Invalid):adapter('cholesky_explicit_v1',MMSETask(),16)
    def test_cholesky_independent_reference_multiple_shapes(self):
        for nt in (8,16):
            t=MMSETask(batch_size=24,nr=32,nt=nt,nrhs=4);f=fixture(t,123)
            r=simulate_cholesky(f['H'],f['Y'],.1,1)
            np.testing.assert_allclose(r['output'],task_reference(f,t),rtol=1e-12,atol=1e-12)
            self.assertNotIn('X0',r)
    def test_modeled_reference_task_quality(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,42)
        r=simulate_cholesky_fp16_model(f['H'],f['Y'],.1,1)
        self.assertLess(np.linalg.norm(r['output']-task_reference(f,t))/np.linalg.norm(task_reference(f,t)),.02)
    def test_effort_overlaps_union_and_thinking_unknown(self):
        r=summarize(0,10_000_000_000,[dict(start_ns=1_000_000_000,end_ns=6_000_000_000),dict(start_ns=4_000_000_000,end_ns=8_000_000_000)])
        self.assertEqual(r['observed_tool_union_seconds'],7);self.assertEqual(r['agent_or_uninstrumented_seconds'],3);self.assertIsNone(r['pure_model_thinking_seconds'])
    def test_effort_rejects_bad_clock(self):
        with self.assertRaises(ValueError):summarize(2,1,[])

if __name__=='__main__':unittest.main()
