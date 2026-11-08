import unittest
import numpy as np
from verification.studies.bri_math import system,iterate,enclosure,check_bounds,root_product,relative,store
from verification.contracts.core import Invalid

class BRIPrinciplesTests(unittest.TestCase):
    def matrices(self):
        rng=np.random.default_rng(101);H=rng.normal(size=(3,8,4))+1j*rng.normal(size=(3,8,4));return system(H,.1,2,'fp64')
    def test_scalar_exposes_literal_formula_wrong_target(self):
        A=np.array([[[2.+0j]]]);P=np.array([[[.5+0j]]]);ev=np.array([[1.]])
        correct=list(iterate(A,P,ev,30,method='optimal'))[-1][1]
        wrong=list(iterate(A,P,ev,30,method='optimal',form='literal_P'))[-1][1]
        np.testing.assert_allclose(correct,.5,atol=1e-12);np.testing.assert_allclose(wrong,2.,atol=1e-12)
    def test_scaled_form_and_right_recovery_equal_direct(self):
        A,P,ev=self.matrices();a=list(iterate(A,P,ev,16));b=list(iterate(A,P,ev,16,form='scaled'))
        for (_,x),(_,y) in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-12)
        self.assertGreater(np.linalg.norm(P@A-A@P),.1)
    def test_chebyshev_three_term_matches_independent_root_product(self):
        A,P,ev=self.matrices();lo,hi=enclosure(ev)
        for K in (1,2,4,8):
            x=list(iterate(A,P,ev,K))[-1][1];y=root_product(A,P,K,lo,hi)
            np.testing.assert_allclose(x,y,rtol=1e-10,atol=1e-11)
    def test_full_block_reduces_to_direct_inverse(self):
        rng=np.random.default_rng(102);H=rng.normal(size=(2,8,4))+1j*rng.normal(size=(2,8,4));A,P,ev=system(H,.1,4,'fp64')
        x=list(iterate(A,P,ev,1))[-1][1];np.testing.assert_allclose(x,np.linalg.inv(A),rtol=1e-12,atol=1e-12)
    def test_spectrum_similarity_and_bounds_refuse_outliers(self):
        A,P,ev=self.matrices()
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(P@A).real,axis=-1),ev,rtol=1e-11)
        lo,hi=enclosure(ev);check_bounds(ev,lo,hi)
        with self.assertRaises(Invalid):check_bounds(ev,lo,ev[:,-1]*.9)
        with self.assertRaises(Invalid):check_bounds(ev,ev[:,0]*1.1,hi)
    def test_stationary_error_propagation_and_divergence(self):
        A=np.array([np.diag([1.,5.])],complex);P=np.eye(2)[None].astype(complex);ev=np.array([[1.,5.]])
        xstar=np.linalg.inv(A);X=list(iterate(A,P,ev,8,method='half'))[-1][1]
        prediction=np.linalg.matrix_power(np.eye(2)-.5*A[0],8)@(-xstar[0])
        np.testing.assert_allclose(X[0]-xstar[0],prediction)
        self.assertGreater(relative(X,xstar)[0],1)
    def test_no_finite_clipping_and_explicit_precision_types(self):
        self.assertEqual(store([1.+2j],'fp32').dtype,np.complex64)
        self.assertFalse(np.isfinite(store([1e10+0j],'half_storage')).all())
        with self.assertRaises(Invalid):store([1.],'fp16_hardware')

    def test_campaign_measurements_are_strict_json_serializable(self):
        import json
        from verification.studies.bri_campaign import measure
        A,P,ev=self.matrices();X=np.linalg.inv(A);rhs=np.ones((3,4,2),complex);truth=X@rhs
        bits=np.ones((3,4,2,2),dtype=np.uint8)
        r=measure(X,A,X,A,X,rhs,truth,rhs,'fp64',bits)
        json.dumps(r,allow_nan=False)
        self.assertIs(type(r['passed']),bool)

    def test_explicit_hermitian_storage_is_separate_precision_design(self):
        rng=np.random.default_rng(103);H=rng.normal(size=(12,64,16))+1j*rng.normal(size=(12,64,16))
        A,P,_=system(H,.1,4,'half_storage',hermitian_storage=True)
        np.testing.assert_array_equal(A,A.conj().transpose(0,2,1))
        np.testing.assert_array_equal(P,P.conj().transpose(0,2,1))
        self.assertEqual(A.dtype,np.complex64)
