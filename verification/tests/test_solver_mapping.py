import unittest
import numpy as np
from verification.contracts.mmse import MMSETask,fixture,task_reference
from verification.contracts.core import Invalid
from verification.operators.catalog import adapter
from verification.reference.ldl import simulate_ldl,simulate_ldl_fp16_model
from verification.reference.bri import simulate_bri,simulate_bri_fp16_model

class SolverMappingTests(unittest.TestCase):
    def test_ldl_unit_diagonal_and_row_weights(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,123);r=simulate_ldl(f['H'],f['Y'],.1,1)
        np.testing.assert_allclose(r['output'],task_reference(f,t),atol=1e-12)
        np.testing.assert_array_equal(np.diagonal(r['evidence']['factor_L'],axis1=1,axis2=2),np.ones((24,8)))
        for h,l,d in zip(f['H'],r['evidence']['factor_L'],r['evidence']['factor_D']):np.testing.assert_allclose((l*d)@l.conj().T,h.conj().T@h+.1*np.eye(8),atol=1e-12)
    def test_ldl_modeled_quality(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,42);r=simulate_ldl_fp16_model(f['H'],f['Y'],.1,1)
        self.assertLess(np.linalg.norm(r['output']-task_reference(f,t))/np.linalg.norm(task_reference(f,t)),.02)
        np.testing.assert_array_equal(np.diagonal(r['evidence']['factor_L_block'],axis1=1,axis2=2),np.ones((24,16)))
    def test_bri_reference_convergence_and_zero_offblock(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,123);r=simulate_bri(f['H'],f['Y'],.1,192)
        np.testing.assert_allclose(r['output'],task_reference(f,t),rtol=1e-10,atol=1e-10)
        mask=np.indices((16,16));off=mask[0]//2!=mask[1]//2;self.assertTrue((r['evidence']['preconditioner_block'][:,off]==0).all())
        low=simulate_bri(f['H'],f['Y'],.1,1);self.assertGreater(np.linalg.norm(low['output']-task_reference(f,t))/np.linalg.norm(task_reference(f,t)),.02)
    def test_bri_modeled_reference_quality(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,42);r=simulate_bri_fp16_model(f['H'],f['Y'],.1,192)
        self.assertLess(np.linalg.norm(r['output']-task_reference(f,t))/np.linalg.norm(task_reference(f,t)),.02)
    def test_iteration_limits_are_per_adapter(self):
        e,p,_,_=adapter('bri_explicit_v1',MMSETask(),192);self.assertEqual(p['iterations'],192)
        for name,k in [('ns_corrected_v1',192),('bri_explicit_v1',257),('ldl_explicit_v1',16)]:
            with self.assertRaises(Invalid):adapter(name,MMSETask(),k)

if __name__=='__main__':unittest.main()
