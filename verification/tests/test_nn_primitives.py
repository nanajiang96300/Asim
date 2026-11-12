import copy,unittest
import numpy as np
from verification.contracts.core import Invalid,Unsupported
from verification.profiles.numeric import policy_for,bit_equal,F16
from verification.reconstruction.engine import arithmetic
from verification.primitives.nn import validate_instruction
from verification.reference.tiny_nn import simulate
from verification.tasks.tiny_nn import NNTask,fixture,graph,hardware_check

class NNPrimitiveTests(unittest.TestCase):
    def test_capacity_config_is_explicit_and_layout_fits(self):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[2]
        good=json.loads((root/'configs/ascend_910b_nn_v1.json').read_text());hardware_check(good)
        with self.assertRaises(Invalid):hardware_check(json.loads((root/'configs/ascend_910b_quiet.json').read_text()))
        g=graph(NNTask(length=32,hidden=64,ffn=128))
        self.assertEqual(sum(x['shape'][0]*x['shape'][1]*2 for x in g['inputs'])+32768,105344)
        self.assertLess(sum(x['shape'][0]*x['shape'][1]*2 for x in g['nodes']),131072)
    def test_bad_or_missing_intermediate_not_hidden_by_correct_output(self):
        from verification.reconstruction.runner import compare_stage_captures
        wanted=np.ones((1,2,3),np.float16);correct_final=np.ones((2,3),np.float16)
        checks=compare_stage_captures({'Y':{'index':7,'value':correct_final},'P':{'index':3,'value':np.zeros((2,3))}}, {'Y','P'}, {'Y':wanted,'P':wanted},0)
        self.assertTrue(next(c for c in checks if c['name']=='Y')['bit_equal'])
        self.assertFalse(all(c['bit_equal'] for c in checks))
        self.assertFalse(compare_stage_captures({}, {'P'}, {'P':wanted},0)[0]['bit_equal'])
    def test_mixed_storage_metrics_do_not_invent_quantization_error(self):
        from verification.shadow.metrics import relative_error
        x=np.array([[1.1,-1.2,2.3]],np.float16)
        self.assertEqual(relative_error(x.astype(np.float64),x),0)
        self.assertEqual(relative_error(x,x.astype(np.float64)),0)
        bad=x.astype(np.float64);bad[0,1]+=.1
        self.assertGreater(relative_error(bad,x),0)
    def test_old_profile_frozen_unchanged(self):
        self.assertNotIn('nonlinear_contract',F16.frozen());self.assertEqual(F16.task_tolerance,.02)
        self.assertTrue(policy_for('modeled_fp16_fp32_nn_v1').modeled)
        with self.assertRaises(Unsupported):arithmetic('SOFTMAX',[np.zeros((2,3))],precision=F16)
    def test_softmax_non_square_stable_and_axis(self):
        x=np.array([[1000,1000,1000],[-1000,0,1000]],np.float32)
        p=arithmetic('SOFTMAX',[x],precision=policy_for('modeled_fp16_fp32_nn_v1'))
        np.testing.assert_allclose(p[0],np.ones(3)/3,rtol=1e-6);np.testing.assert_array_equal(p[1],[0,0,1]);self.assertTrue(np.isfinite(p).all())
    def test_gelu_signed_and_nontrivial(self):
        x=np.array([[-8,-1,-.5,-0.,0.,.5,1.,8]],np.float32)
        y=arithmetic('GELU',[x],precision=policy_for('modeled_fp16_fp32_nn_v1'))
        self.assertLess(y[0,1],0);self.assertGreater(y[0,6],.8);self.assertTrue(np.signbit(y[0,3]));self.assertLess(y[0,2],0)
    def test_explicit_row_broadcast_only(self):
        p=policy_for('modeled_fp16_fp32_nn_v1');x=np.ones((3,4));b=np.arange(4)[None,:]
        with self.assertRaises(Invalid):arithmetic('ADD',[x,b],precision=p)
        np.testing.assert_array_equal(arithmetic('ADD',[x,b],'row_rhs',p),x+b)
        with self.assertRaises(Invalid):arithmetic('ADD',[x,np.ones((2,4))],'row_rhs',p)
    def test_workload_axis_version_and_formula_rejected(self):
        v=dict(base=0x10000000,row=0,col=0,rows=16,cols=32,transpose=False,conjugate=False)
        i=dict(opcode=8,m=16,k=32,n=32,compute_size=128,size=16,semantic=dict(src=[v],dst=v,kernel_contract='nn-vector-v1',axis=-1))
        validate_instruction(i,'modeled_fp16_fp32_nn_v1')
        for path,value in [('axis',0),('kernel_contract','bad')]:
            z=copy.deepcopy(i);z['semantic'][path]=value
            with self.assertRaises(Unsupported):validate_instruction(z,'modeled_fp16_fp32_nn_v1')
        z=copy.deepcopy(i);z['compute_size']=64
        with self.assertRaises(Invalid):validate_instruction(z,'modeled_fp16_fp32_nn_v1')
        z=copy.deepcopy(i);z['opcode']=22;z['compute_size']=16*32*4;z['semantic']['formula']='erf'
        with self.assertRaises(Unsupported):validate_instruction(z,'modeled_fp16_fp32_nn_v1')
    def test_independent_reference_primitive_bit_agreement(self):
        p=policy_for('modeled_fp16_fp32_nn_v1')
        for kind in ['softmax','gelu']:
            t=NNTask(kind=kind);d=fixture(t,111);ref=simulate(d['inputs'],t)['output']
            got=np.stack([p.store(arithmetic(kind.upper(),[x],precision=p)) for x in d['inputs']['X']]);self.assertTrue(bit_equal(got,ref))
    def test_task_has_real_projection_and_parameters(self):
        g=graph(NNTask());self.assertEqual(sum(n['op']=='GEMM' for n in g['nodes']),8)
        self.assertIn('scale',[x['name'] for x in g['inputs']]);self.assertEqual(sum(n['op']=='SOFTMAX' for n in g['nodes']),1)
        self.assertNotEqual(graph(NNTask(kind='mlp')),g)
if __name__=='__main__':unittest.main()
