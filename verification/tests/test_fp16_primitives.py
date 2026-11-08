import unittest
import numpy as np
from verification.studies.fp16_primitives import Primitives,precondition,block,run
from verification.profiles.numeric import F16
from verification.contracts.core import Invalid,Unsupported

class FP16PrimitiveTests(unittest.TestCase):
    def test_batched_gemm_and_vectors_match_frozen_policy_bit_for_bit(self):
        rng=np.random.default_rng(144);e=Primitives();a=e.const(rng.normal(size=(3,4,7)));b=e.const(rng.normal(size=(3,7,2)))
        x=e.op('GEMM',a,b);expected=np.stack([F16.store(F16.calculate('GEMM',[v,w])) for v,w in zip(a,b)])
        np.testing.assert_array_equal(x.view(np.uint16),expected.view(np.uint16))
        for code in ['ADD','SUB','MUL','DIV']:
            v=e.const(np.full_like(x,.75));np.testing.assert_array_equal(e.op(code,x,v),F16.store(F16.calculate(code,[x,v])))
    def test_no_fp32_state_or_nonfinite_repair(self):
        e=Primitives()
        with self.assertRaises(Invalid):e.op('ADD',np.ones((2,2),np.float32),np.ones((2,2),np.float32))
        with self.assertRaises((Invalid,FloatingPointError)):e.const(np.array([1e10]))
        with self.assertRaises(Unsupported):e.op('MAGIC',e.const(np.ones((1,1))))
    def test_real_block_preconditioner_uses_only_primitives(self):
        rng=np.random.default_rng(145);H=rng.normal(size=(2,8,4))+1j*rng.normal(size=(2,8,4));e=Primitives('float64');h=e.const(block(H));l=e.const(np.full((2,1,1),.2));P=precondition(e,h,l,2)
        A=h.transpose(0,2,1)@h+.2*np.eye(8)
        for idx in [[0,1,4,5],[2,3,6,7]]:
            np.testing.assert_allclose(P[:,idx][:,:,idx],np.linalg.inv(A[:,idx][:,:,idx]),rtol=1e-12,atol=1e-12)
    def test_factored_and_explicit_inverse_equal_in_exact_arithmetic(self):
        rng=np.random.default_rng(146);H=(rng.normal(size=(2,8,4))+1j*rng.normal(size=(2,8,4)))/5;Y=rng.normal(size=(2,8,2))+1j*rng.normal(size=(2,8,2))
        a=list(run(H,Y,.1,2,'explicit_inverse',12,storage='float64'));b=list(run(H,Y,.1,2,'implicit_inverse',12,storage='float64'))
        for x,y in zip(a,b):np.testing.assert_allclose(x[1],y[1],rtol=1e-11,atol=1e-11)
        self.assertEqual(list(run(H,Y,.1,2,'implicit_solve',2))[-1][1].dtype,np.float16)

    def test_single_sided_representation_and_half_compensation_are_explicit(self):
        rng=np.random.default_rng(147);H=(rng.normal(size=(2,8,4))+1j*rng.normal(size=(2,8,4)))/6;Y=rng.normal(size=(2,8,2))+1j*rng.normal(size=(2,8,2))
        x=list(run(H,Y,.1,1,'implicit_solve',8,left=True,compensated=True,restart=4))[-1]
        self.assertEqual(x[1].dtype,np.float16);self.assertEqual(x[2]['compensation_storage'],'float16')
        np.testing.assert_array_equal(x[1][:,:4,:2],x[1][:,4:,2:])
        np.testing.assert_array_equal(x[1][:,:4,2:],-x[1][:,4:,:2])
        a=list(run(H,Y,.1,2,'implicit_inverse',8,storage='float64',left=True,hermitian=True))[-1][1]
        b=list(run(H,Y,.1,2,'implicit_inverse',8,storage='float64'))[-1][1]
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
