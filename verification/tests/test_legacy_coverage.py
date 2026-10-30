import copy,unittest
import numpy as np
from verification.tests.test_legacy_replay import fixture,v
from verification.legacy.coverage import audit
from verification.legacy.stages import ldl_reference,compare
from verification.reconstruction.source_views import execute_batch

class CoverageTests(unittest.TestCase):
    def test_scan_after_ambiguity(self):
        h,ops,inputs=fixture();ops[2]['semantic']={'schema':'source-views-v1','opcode':ops[2]['opcode'],'invalid_reason':'unknown reduction'}
        ops[3]['semantic']['dst']['row']=999
        a=audit(h,ops,inputs,0,(1,1))
        self.assertEqual(a['counts']['instructions'],len(ops))
        self.assertIn('SEMANTIC_AMBIGUITY',a['counts']);self.assertIn('BOUNDS_ERROR',a['counts'])
    def test_snapshot_no_synthetic_store_and_no_resume(self):
        h,ops,inputs=fixture();ops[-1]['semantic']={'schema':'source-views-v1','opcode':1,'invalid_reason':'unknown'}
        a=execute_batch(h,ops,inputs,0,np.array([[7.1]]),captures={2:{'gram':ops[2]['semantic']['dst']},len(ops)-1:{'late':v(0)}})
        self.assertEqual(a['status'],'SEMANTIC_AMBIGUITY');self.assertEqual(a['_observations']['gram']['status'],'CAPTURED')
        self.assertNotIn('late',a['_observations'])
    def test_snapshot_is_copy_and_bad_observation_does_not_change_primary(self):
        h,ops,inputs=fixture();a=execute_batch(h,ops,inputs,0,np.array([[7.1]]),captures={2:{'bad':v(0)}})
        b=execute_batch(h,ops,inputs,0,np.array([[7.1]]))
        self.assertEqual(a['status'],b['status']);self.assertEqual(a['_observations']['bad']['status'],'VIEW_INVALID')
    def test_no_implicit_broadcast(self):
        h,ops,inputs=fixture();ops[2]['semantic']['src'][0]['rows']=1
        self.assertIn('SHAPE_MISMATCH',audit(h,ops,inputs,0,(1,1))['counts'])
    def test_independent_ldl_reference(self):
        rng=np.random.default_rng(42);h=rng.normal(size=(8,4))+1j*rng.normal(size=(8,4));a=h.conj().T@h+.1*np.eye(4)
        l,d=ldl_reference(a);np.testing.assert_allclose((l*d)@l.conj().T,a,atol=1e-12)
        z=np.linalg.solve(l,np.eye(4));np.testing.assert_allclose(z.conj().T@np.diag(1/d)@z,np.linalg.inv(a),atol=1e-12)
        wrong=(z*np.sqrt(1/d));self.assertGreater(np.linalg.norm(wrong.conj().T@wrong-np.linalg.inv(a)),1e-4)
    def test_not_reached_is_not_stage_pass(self):
        self.assertEqual(compare({},dict(x=np.eye(2)))['x']['status'],'NOT_REACHED')

if __name__=='__main__':unittest.main()
