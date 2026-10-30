import unittest
import numpy as np
from verification.reconstruction.engine import ArrayMemory,full_view,execute_views,ReplayFault,arithmetic
from verification.contracts.core import Invalid,Unsupported
from verification.contracts.mmse import MMSETask,fixture,task_reference,decode_checked,assert_comparable
from verification.reference.ns import simulate_ns

class UnifiedTests(unittest.TestCase):
    def test_partial_initialized_shared_memory(self):
        m=ArrayMemory({'a':dict(rows=2,cols=2,name='a')});v=full_view('a',(1,1));m.write(v,np.ones((1,1)))
        with self.assertRaises(ReplayFault):m.read(full_view('a',(2,2)))
    def test_broadcast_only_explicit(self):
        a=np.ones((2,2));b=np.array([[2.]])
        with self.assertRaises(ReplayFault):arithmetic('MUL',[a,b])
        np.testing.assert_array_equal(arithmetic('MUL',[a,b],'scalar_rhs'),a*2)
    def test_unknown_primitive_no_fallback(self):
        with self.assertRaises(Unsupported):arithmetic('inverse',[np.eye(2)])
    def test_scope_prevents_comparison(self):
        a=MMSETask();b=MMSETask(scope='prepared_solver_kernel_v1')
        with self.assertRaises(Invalid):assert_comparable(a.frozen(),b.frozen())
    def test_future_channel_no_random_fallback(self):
        with self.assertRaises(Unsupported):MMSETask(channel='cdl-b').validate()
    def test_candidate_inputs_not_precomputed_solver(self):
        f=fixture(MMSETask(),42)
        self.assertEqual(set(f['inputs']),{'H','Y','RegI','Identity'})
    def test_independent_reference_initialization_and_lowK(self):
        t=MMSETask();f=fixture(t,42);a=f['H'][0].conj().T@f['H'][0]+t.regularization*np.eye(t.nt)
        r=simulate_ns(f['H'],f['Y'],t.regularization,1)
        np.testing.assert_allclose(r['X0'][0],np.eye(t.nt)/(2*np.trace(a).real))
        self.assertGreater(np.linalg.norm(r['output']-task_reference(f,t)),1.)
        r=simulate_ns(f['H'],f['Y'],t.regularization,16)
        np.testing.assert_allclose(r['output'],task_reference(f,t),atol=1e-11,rtol=1e-11)
    def test_decode_must_check_all_quadrants(self):
        z=np.zeros((4,2));z[2,1]=1
        with self.assertRaises(Invalid):decode_checked(z,2,1)
    def test_generic_SUB_not_raw_ADD(self):
        from verification.tests.test_asim_adapter import fixture,exported
        from verification.asim.runner import recover
        p=fixture();p['ops'].insert(1,{'id':'subtract','opcode':'SUB','src':['x','x'],'dst':'x'})
        r=exported(fixture());r['header']={k:v for k,v in p.items() if k!='ops'}
        r['instructions'].insert(2,dict(r['instructions'][0],id='subtract',opcode=10,semantic=p['ops'][1]))
        with self.assertRaises(Unsupported):recover(r,p)

    def test_scalar_trace_must_follow_dispatch(self):
        from verification.asim.batch import check_events
        event={'name':'scalar','unit':'Core0_Scalar','start_cycle':'1','end_cycle':'2'}
        dispatch={0:{'ids':['scalar'],'cycle':3}}
        with self.assertRaises(Invalid):check_events([event],{'scalar':'Core0_Scalar'},{},10,dispatch)
