import unittest
import numpy as np
from verification.contracts.mmse import MMSETask,fixture,task_reference
from verification.reference.structured import simulate_cholesky_block,simulate_cholesky_block_fp16,simulate_ldl_block,simulate_ldl_block_fp16,simulate_merge,simulate_merge_fp16
from verification.operators.catalog import adapter
from verification.contracts.core import Invalid,Unsupported

class StructuredTests(unittest.TestCase):
    def test_independent_block_and_merge_references(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4);f=fixture(t,42);target=task_reference(f,t)
        for fn in [simulate_cholesky_block,simulate_ldl_block,simulate_merge]:
            r=fn(f['H'],f['Y'],.1,1);np.testing.assert_allclose(r['output'],target,rtol=1e-10,atol=1e-10)
            for b,h in enumerate(f['H']):
                hreal=np.block([[h.real,-h.imag],[h.imag,h.real]]);a=hreal.T@hreal+.1*np.eye(16);l=r['evidence']['factor_L_block'][b]
                reconstructed=(l*r['evidence']['factor_D_block'][b])@l.T if 'factor_D_block' in r['evidence'] else l@l.T
                np.testing.assert_allclose(reconstructed,a,rtol=1e-12,atol=1e-12)
    def test_independent_fp16_quality_two_seeds(self):
        t=MMSETask(batch_size=24,nr=32,nt=8,nrhs=4)
        for seed in (42,123):
            f=fixture(t,seed);target=task_reference(f,t)
            for fn in [simulate_cholesky_block_fp16,simulate_ldl_block_fp16,simulate_merge_fp16]:
                r=fn(f['H'],f['Y'],.1,1);self.assertLess(np.linalg.norm(r['output']-target)/np.linalg.norm(target),.02)
    def test_adapter_parameters_are_explicit(self):
        for name in ('cholesky_block','ldl_block','cholesky_merge'):
            _,p,_,_=adapter(name+'_explicit_v1',MMSETask(),1);self.assertIn('block_size_real',p);self.assertNotIn('iterations',p)
            with self.assertRaises(Invalid):adapter(name+'_explicit_v1',MMSETask(),16)
        with self.assertRaises(Unsupported):adapter('unknown_merge',MMSETask(),1)

if __name__=='__main__':unittest.main()
