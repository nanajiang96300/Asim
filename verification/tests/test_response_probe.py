import copy
import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from verification.communication.response import identify,linear_rate
from verification.communication.probe import validate,evaluate,ActualResponse
from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask
from verification.profiles.numeric import F16
from verification.reconstruction.daily import sha

REPO=Path(__file__).resolve().parents[2]
LIMITS=dict(max_prediction_error=.01,max_complex_leakage=.01,max_cross_rhs_leakage=.01,max_zero_offset=0.,max_rate_loss=.05)


def inputs():
    rng=np.random.default_rng(33)
    H=(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))/3
    Y=(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))
    W=np.linalg.solve(H.conj().T@H+.1*np.eye(2),H.conj().T)
    return H,Y,W


def contract(path='source.json'):
    return dict(schema='actual-response-se-proxy-v1',runtime=dict(path='runtime.json',sha256='0'*64),
                cases=[dict(path=path,sha256='0'*64,role='historical_diagnostic')],batch_indices=[0],
                probe_seed=1,max_replay_calls=100,limits=dict(LIMITS))


class ResponseTests(unittest.TestCase):
    def test_identity_rate_and_row_scale_invariance(self):
        a=linear_rate(np.eye(2),np.eye(2),1.)
        self.assertAlmostEqual(a['sum_bits_per_s_per_hz'],2.)
        b=linear_rate(np.diag([.25,2j]),np.eye(2),1.)
        np.testing.assert_allclose(a['sinr_per_stream'],b['sinr_per_stream'])

    def test_filtered_noise_and_interference(self):
        w=np.array([[1.,1.],[0.,1.]])
        r=linear_rate(w,np.eye(2),.5)
        np.testing.assert_allclose(r['sinr_per_stream'],[.5,2.])
        np.testing.assert_allclose(r['filtered_noise_power'],[1.,.5])
        with self.assertRaises(Invalid):linear_rate(np.zeros((2,2)),np.eye(2),.1)

    def test_basis_recovers_complex_map(self):
        h,y,w=inputs();r,a=identify(lambda name,q:w@q,h,y,.1,1,LIMITS)
        self.assertEqual(r['status'],'SE_PROXY_SAMPLE_CHECKED')
        np.testing.assert_allclose(a['W'],w,atol=1e-14)
        np.testing.assert_allclose(a['V'],0,atol=1e-14)
        self.assertEqual(r['replay_calls'],13)
        self.assertAlmostEqual(r['se_loss_bits_per_s_per_hz'],0,places=12)

    def test_widely_linear_map_not_silently_accepted(self):
        h,y,w=inputs();r,a=identify(lambda name,q:w@q+.2*w@q.conj(),h,y,.1,1,LIMITS)
        self.assertEqual(r['status'],'RESPONSE_MODEL_REJECTED')
        self.assertGreater(r['complex_leakage'],.1)
        self.assertNotIn('candidate_linear_se_proxy',r)

    def test_nonlinearity_and_offset_rejected(self):
        h,y,w=inputs()
        for f in [lambda name,q:w@(q*np.abs(q)),lambda name,q:w@q+.01]:
            r,_=identify(f,h,y,.1,1,LIMITS)
            self.assertEqual(r['status'],'RESPONSE_MODEL_REJECTED')

    def test_rhs_mixing_rejected(self):
        h,y,w=inputs();r,_=identify(lambda name,q:w@q[:,::-1],h,y,.1,1,LIMITS)
        self.assertEqual(r['status'],'RESPONSE_MODEL_REJECTED')
        self.assertGreater(r['cross_rhs_leakage'],.9)

    def test_rate_loss_is_additional_gate(self):
        h,y,w=inputs();bad=np.array([[1.,0,0,0],[1.,0,0,0]],complex)
        r,_=identify(lambda name,q:bad@q,h,y,.1,1,LIMITS)
        self.assertEqual(r['status'],'SE_PROXY_QUALITY_FAILED')
        self.assertGreater(r['se_loss_bits_per_s_per_hz'],.05)

    def test_missing_or_nonfinite_probe_output_rejected(self):
        h,y,w=inputs()
        for f in [lambda name,q:np.zeros((1,1)),lambda name,q:w@q*np.nan]:
            with self.assertRaises(Invalid):identify(f,h,y,.1,1,LIMITS)

    def test_probe_contract_fail_closed(self):
        for key,value in [('batch_indices',[0,0]),('probe_seed',True),('max_replay_calls',2049),('extra',1)]:
            c=contract();c[key]=value
            with self.assertRaises(Invalid):validate(c)
        c=contract();c['limits']['max_prediction_error']=float('nan')
        with self.assertRaises(Invalid):validate(c)

    def test_budget_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.json';p.write_text(json.dumps(dict(task=MMSETask().frozen())))
            rp=root/'runtime.json';rp.write_text('{}')
            c=contract(str(p));c['cases'][0]['sha256']=sha(p);c['runtime']=dict(path=str(rp),sha256=sha(rp));c['max_replay_calls']=1
            rt=dict(python_version=platform.python_version(),numpy_version=np.__version__)
            with patch('verification.communication.probe.load_runtime',return_value=rt),patch('verification.communication.probe.load_case') as intake:
                with self.assertRaisesRegex(Invalid,'budget exceeded'):evaluate(c,REPO,root)
                intake.assert_not_called()

    def response_fixture(self):
        x=object.__new__(ActualResponse);x.task=MMSETask(batch_size=24,nr=4,nt=2,nrhs=2);x.batch=0
        x.data=dict(inputs=dict(Y=np.zeros((24,8,4),np.float16)));x.header=dict(inputs=[dict(name='source.Y')])
        x.reference=lambda *args:dict(output_block=np.zeros((1,4,4)))
        x.H=np.ones((4,2),complex);x.result=dict(algorithm={});x.groups={0:[]};x.policy=F16
        x.original=np.zeros((4,4));x.arrays={};x.ledger=[]
        return x

    def test_probe_cannot_ignore_interpreter_or_oracle_failure(self):
        for status,value in [('NUMERIC_FAILED',np.zeros((4,4))),('STRUCTURE_MATCH',np.ones((4,4)))]:
            x=self.response_fixture()
            fake=dict(status=status,executed=1,total=1,cost_issues=[],_output=value)
            with patch('verification.communication.probe.execute_batch',return_value=fake):
                with self.assertRaisesRegex(Invalid,'reference mismatch'):x('original',np.ones((4,2)))

    def test_normal_payload_must_reproduce_sealed_actual_output(self):
        x=self.response_fixture();x.original=np.ones((4,4))
        fake=dict(status='STRUCTURE_MATCH',executed=1,total=1,cost_issues=[],_output=np.zeros((4,4)))
        with patch('verification.communication.probe.execute_batch',return_value=fake):
            with self.assertRaisesRegex(Invalid,'archived output'):x('original',np.ones((4,2)))


if __name__=='__main__':unittest.main()
