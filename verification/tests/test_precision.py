import unittest
import numpy as np
from verification.profiles.numeric import F16,F64,policy_for,bit_equal
from verification.contracts.core import Invalid,Unsupported
from verification.reconstruction.engine import ArrayMemory,full_view,execute_views

class PrecisionTests(unittest.TestCase):
    def test_round_nearest_even_positive_negative_ties(self):
        x=np.array([1+2**-11,1+3*2**-11,-1-2**-11,-1-3*2**-11])
        np.testing.assert_array_equal(F16.store(x),[1,1+2**-9,-1,-1-2**-9])
    def test_signed_zero_and_half_subnormal(self):
        q=F16.store(np.array([0.,-0.,2**-24,2**-25,3*2**-25]))
        self.assertTrue(np.signbit(q[1]));self.assertEqual(float(q[2]),2**-24)
        self.assertEqual(float(q[3]),0.);self.assertEqual(float(q[4]),2**-23)
        self.assertFalse(bit_equal(np.array([0.]),np.array([-0.])))
    def test_fp32_subnormal_not_flushed(self):
        v=F16.calculate('MUL',[np.array([[2**-70]]),np.array([[2**-70]])])
        self.assertEqual(float(v[0,0]),2**-140)
    def test_writeback_overflow(self):
        with self.assertRaises(Invalid):F16.store(np.array([70000.]))
    def test_fp32_intermediate_overflow(self):
        with self.assertRaises(FloatingPointError):F16.calculate('MUL',[np.array([[1e30]]),np.array([[1e30]])])
    def test_ordered_gemm_cancellation_differs_from_fp64(self):
        a=np.array([[2048.,.5,-2048.]]);b=np.ones((3,1))
        # All terms exact here. The result is accumulated in f32, then half.
        self.assertEqual(float(F16.calculate('GEMM',[a,b])[0,0]),.5)
        x=np.array([[2**24,1.,-2**24]])
        self.assertEqual(float(F16.calculate('GEMM',[x,b])[0,0]),0.)
        self.assertEqual(float((x@b)[0,0]),1.)
    def test_each_instruction_writes_half(self):
        mem=ArrayMemory({'x':dict(rows=1,cols=1,name='x'),'y':dict(rows=1,cols=1,name='y')},np.float64,F16)
        x=full_view('x',(1,1));y=full_view('y',(1,1));mem.write(x,np.array([[1.]]));mem.write(y,np.array([[2**-11]]))
        execute_views(mem,'ADD',[x,y],x);execute_views(mem,'ADD',[x,y],x)
        self.assertEqual(float(mem.read(x)[0,0]),1.)
    def test_bit_gate_not_norm_gate(self):
        a=np.array([100.,2**-24]);b=np.array([100.,0.])
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(a),1e-8)
        self.assertFalse(bit_equal(a,b))
    def test_unknown_policy_not_guessed(self):
        with self.assertRaises(Unsupported):policy_for('hardware_fp16_fused')
    def test_half_reference_metric_uses_wide_comparison(self):
        from verification.tests.test_legacy_replay import fixture
        from verification.reconstruction.source_views import execute_batch
        h,ops,inputs=fixture();h['schema']='source-views-program-v1';h['profile']=F16.name
        inputs=[a.real.astype(np.float16) for a in inputs]
        # Real H=[1,2] gives 5+rounded(.1), finally rounded to half.
        r=execute_batch(h,ops,inputs,0,np.array([[5.1]],dtype=np.float16))
        self.assertEqual(r['status'],'STRUCTURE_MATCH')

    def test_half_bit_gate_rejects_hidden_extra_precision(self):
        self.assertFalse(bit_equal(np.array([1.+1e-12]),np.array([1.])))
    def test_model_gemm_inner_shape_not_truncated(self):
        from verification.reconstruction.engine import arithmetic,ReplayFault
        with self.assertRaises(ReplayFault):arithmetic('GEMM',[np.ones((1,2)),np.ones((3,1))],precision=F16)
