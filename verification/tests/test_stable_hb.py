import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from verification.contracts.core import Unsupported,Invalid
from verification.studies.stable_hb import HeavyBall
from verification.reference.stable_hb import simulate

class FixedHBTests(unittest.TestCase):
    def fixture(self,n=24):
        rng=np.random.default_rng(58200);H=(rng.normal(size=(n,64,16))+1j*rng.normal(size=(n,64,16)))/np.sqrt(128)
        H*=np.sqrt(16/np.sum(abs(H)**2,axis=(1,2)))[:,None,None]
        Y=rng.normal(size=(n,64,8))+1j*rng.normal(size=(n,64,8));return H,Y
    def test_independent_bits_and_setup(self):
        H,Y=self.fixture();a=HeavyBall(H,Y,.01)
        for _,v,_ in a.iterate(10):pass
        b=simulate(H,Y,.01,10)
        self.assertTrue(np.array_equal(v.view(np.uint16),b['output_block'].view(np.uint16)))
        for k,s in a.setup.items():self.assertTrue(np.array_equal(s.reshape(len(H),-1).view(np.uint16),b['setup'][k].reshape(len(H),-1).view(np.uint16)),k)
    def test_bounds_enclose_quantized_system_and_constant_roots(self):
        H,Y=self.fixture(3)
        for lam in (1.,.1,.01):
            a=HeavyBall(H,Y,lam)
            for i in range(len(H)):
                h=a.h[i].astype(float);p=a.setup['P_diagonal'][i,:,0].astype(float);w=np.sqrt(np.tile(p,2))
                eig=np.linalg.eigvalsh(w[:,None]*(h.T@h+float(a.l[i,0,0])*np.eye(32))*w[None,:])
                lo=float(a.lo[i,0,0]);hi=float(a.hi[i,0,0]);alpha=float(a.alpha[i,0,0]);beta=float(a.beta[i,0,0])
                self.assertLessEqual(lo,eig[0]);self.assertGreaterEqual(hi,eig[-1])
                for t in (lo,hi,*eig):self.assertLess(max(abs(np.roots([1,-(1+beta-alpha*t),beta]))),1.)
    def test_reject_outside_domain_without_repair(self):
        H,Y=self.fixture(1)
        with self.assertRaises(Unsupported):HeavyBall(H*20,Y,.1)
        with self.assertRaises(Unsupported):HeavyBall(H,np.zeros((1,64,7)),.1)
    def test_zero_rhs_and_constant_coefficients(self):
        H,Y=self.fixture(1);a=HeavyBall(H,np.zeros_like(Y),.1);initial={k:a.setup[k].copy() for k in ['alpha','beta']}
        for _,v,_ in a.iterate(12):self.assertTrue(np.array_equal(v.view(np.uint16),np.zeros(v.shape,np.uint16)))
        for k in initial:self.assertTrue(np.array_equal(a.setup[k],initial[k]))
    def test_selection_rejects_missing_cases_and_foreign_contract(self):
        from verification.studies.stable_hb_campaign import selection
        c={'lambdas':[1.,.1,.01]};approved='a'*64
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'calibration').mkdir();p=root/'calibration/result.json'
            result=dict(status='CALIBRATION_SELECTED',contract_sha256=approved,cases=[dict(lambda_=l,selected_suffix_K=k) for l,k in [(1.,20),(.1,50),(.01,150)]])
            p.write_text(json.dumps(result));sel=dict(schema='fixed-hb-selection-v1',calibration_result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cases=[dict(lambda_=r['lambda_'],K=r['selected_suffix_K']) for r in result['cases']])
            (root/'selection.json').write_text(json.dumps(sel));selection(c,root,approved)
            for mutation in ({**result,'cases':[]},{**result,'contract_sha256':'b'*64}):
                p.write_text(json.dumps(mutation));sel['calibration_result_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(root/'selection.json').write_text(json.dumps(sel))
                with self.assertRaises(Invalid):selection(c,root,approved)
if __name__=='__main__':unittest.main()
