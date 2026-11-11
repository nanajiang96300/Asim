import unittest,numpy as np
from verification.tasks.affine import AffineTask,fixture,adapter,task_reference
from verification.reference.affine import simulate_affine
from verification.contracts.core import Invalid,Unsupported
class AffineTests(unittest.TestCase):
    def test_fixed_calibration_quality_and_dimensions(self):
        for m,k,n in [(16,16,16),(32,32,32),(32,16,48)]:
            t=AffineTask(m=m,k=k,n=n);d=fixture(t,42);a=simulate_affine(d['inputs'],t)['output'];ref=task_reference(d,t)
            err=np.linalg.norm((a-ref).reshape(96,-1),axis=1)/np.maximum(np.linalg.norm(ref.reshape(96,-1),axis=1),1e-15)
            self.assertLess(float(err.max()),.02);self.assertEqual(list(a.shape),t.output_shape)
    def test_intermediate_half_rounding_is_not_fused(self):
        t=AffineTask(m=16,k=16,n=16);v={k:np.zeros((96,16,16),np.float16) for k in ['A','B','C']};v['A'][:,0,0:2]=1;v['B'][:,0,0]=1;v['B'][:,1,0]=2**-11;v['C'][:,0,0]=-1
        a=simulate_affine(v,t)['output'];self.assertEqual(a[0,0,0],0);self.assertEqual((v['A'].astype(float)@v['B'].astype(float)+v['C'])[0,0,0],2**-11)
    def test_missing_or_unknown_contract_rejected(self):
        for t in [AffineTask(batch_size=24),AffineTask(n=17),AffineTask(profile='unknown')]:
            with self.assertRaises((Invalid,Unsupported)):t.validate()
        with self.assertRaises(Unsupported):adapter('unknown',AffineTask(),1)
        with self.assertRaises(Unsupported):adapter('affine_gemm_v1',AffineTask(),2)
    def test_task_has_no_mmse_or_channel_claim(self):
        frozen=AffineTask().frozen();self.assertEqual(frozen['scope'],'a_b_c_to_affine_z_v1');self.assertNotIn('channel',frozen)
