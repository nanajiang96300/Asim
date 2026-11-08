import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from verification.communication.metrics import cp_upper,outage_interval,paired,summarize
from verification.communication.runner import validate,load_case,REQUIRED_ARTIFACTS,evaluate
from verification.contracts.core import Invalid,Unsupported
from verification.contracts.mmse import MMSETask,fixture,task_reference
from verification.fixtures.rayleigh import real_block
from verification.operators.catalog import adapter
from verification.profiles.numeric import policy_for
from verification.reconstruction.daily import sha
from verification.schedule.runtime import HEADER

REPO=Path(__file__).resolve().parents[2]


def contract():
    return dict(schema='rayleigh-qpsk-quality-v1',policy_level='engineering_trial',
                required_metrics=['paired_uncoded_qpsk_ber','raw_symbol_evm'],
                runtime=dict(path='runtime.json',sha256='0'*64),cases=[dict(path='result.json',sha256='0'*64,role='historical_diagnostic')],
                family_alpha=.05,max_channel_ber_increase=.01,max_channel_evm_increase=.01,
                max_outage_probability=.05,min_independent_channels=96)


class CommunicationTests(unittest.TestCase):
    def test_cp_zero_and_extreme_counts(self):
        self.assertAlmostEqual(cp_upper(0,96,.025),1-.025**(1/96),places=14)
        self.assertEqual(cp_upper(96,96,.025),1.)
        self.assertEqual(outage_interval(0,96,.025)['lower'],0.)
        self.assertGreater(cp_upper(0,24,.01),0.)

    def test_cp_matches_independent_scipy(self):
        try: from scipy.stats import binomtest
        except ImportError: self.skipTest('optional independent SciPy confidence-bound check')
        for n,k in [(2,1),(10,3),(96,0),(96,2),(96,48),(192,1),(192,190)]:
            ref=binomtest(k,n).proportion_ci(confidence_level=.95,method='exact')
            bounds=outage_interval(k,n,.025)
            self.assertAlmostEqual(bounds['lower'],ref.low,places=11)
            self.assertAlmostEqual(bounds['upper'],ref.high,places=11)

    def test_paired_truth_and_phase_error(self):
        truth=np.full((96,2,4),(1+1j)/math.sqrt(2))
        good=paired(truth,truth,truth)
        self.assertTrue(all(r['candidate_ber']==0 and r['candidate_raw_symbol_evm']==0 for r in good))
        wrong=paired(-truth,truth,truth)
        self.assertTrue(all(r['candidate_ber']==1 and r['candidate_raw_symbol_evm']==2 for r in wrong))
        self.assertEqual(summarize([dict(channels=wrong)],contract())['status'],'QUALITY_FAILED')

    def test_scaling_not_hidden_by_ber_or_normalization(self):
        truth=np.full((192,2,4),(1+1j)/math.sqrt(2));rows=paired(.5*truth,truth,truth)
        self.assertTrue(all(r['candidate_ber']==0 and r['evm_increase']==.5 for r in rows))
        self.assertEqual(summarize([dict(channels=rows)],contract())['status'],'QUALITY_FAILED')

    def test_zero_errors_do_not_prove_zero_outage(self):
        truth=np.full((24,2,4),(1+1j)/math.sqrt(2));rows=paired(truth,truth,truth)
        r=summarize([dict(channels=rows)],contract())
        self.assertEqual(r['status'],'INCONCLUSIVE')
        self.assertGreater(r['groups'][0]['outage_bounds']['ber']['upper'],.05)

    def test_family_bounds_not_pooled_across_algorithms(self):
        truth=np.full((96,2,4),(1+1j)/math.sqrt(2));rows=paired(truth,truth,truth)
        one=summarize([dict(channels=rows)],contract())
        many=summarize([dict(channels=rows) for _ in range(7)],contract())
        self.assertEqual(one['status'],'QPSK_QUALITY_CHECKED')
        self.assertEqual(many['status'],'INCONCLUSIVE')

    def test_joint_outage_does_not_hide_disjoint_metric_failures(self):
        truth=np.full((500,2,4),(1+1j)/math.sqrt(2));rows=paired(truth,truth,truth)
        for r in rows[:10]: r['ber_increase']=.02
        for r in rows[10:20]: r['evm_increase']=.02
        r=summarize([dict(channels=rows)],contract())
        b=r['groups'][0]['outage_bounds']
        self.assertLess(b['ber']['upper'],.05);self.assertLess(b['evm']['upper'],.05)
        self.assertGreater(b['joint']['upper'],.05)
        self.assertEqual(r['status'],'INCONCLUSIVE')

    def test_nonfinite_shapes_and_non_qpsk_rejected(self):
        x=np.ones((24,2,4),complex)*(1+1j)/math.sqrt(2)
        for bad in [x[:1],x*np.nan]:
            with self.assertRaises(Invalid): paired(bad,x,x)
        with self.assertRaises(Invalid): paired(x,x,x*2)

    def test_unknown_metrics_and_bad_contract_fail_closed(self):
        for metric in ['SE','BLER','64QAM']:
            c=contract();c['required_metrics'].append(metric)
            with self.assertRaises(Unsupported): validate(c)
        for key,value in [('family_alpha',float('nan')),('min_independent_channels',True),('extra',True)]:
            c=contract();c[key]=value
            with self.assertRaises(Invalid): validate(c)

    def case(self,root):
        # Protocol fixture, not evidence of a real Asim run.
        t=MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1')
        data=fixture(t,77);truth=task_reference(data,t)
        actual=np.stack([real_block(x) for x in truth]).astype(np.float16).astype(float)
        for name in REQUIRED_ARTIFACTS:(root/name).write_text('{}')
        np.save(root/'outputs.npy',actual)
        np.savez(root/'channel.npz',H=data['H'],Y=data['Y'],symbols=data['symbols'])
        np.savez(root/'inputs.npz',**data['inputs'])
        np.savez(root/'references.npz',task=truth,algorithm_block=actual)
        _,params,models,_=adapter('ns_corrected_v1',t,12)
        (root/'models.json').write_text(json.dumps(models))
        binary=root/'binary';binary.write_text('unit fixture')
        receipt=dict(status='COMPLETED',returncode=0,binary_sha256=sha(binary),finish_cycle=100)
        (root/'asim.json').write_text(json.dumps(receipt))
        (root/'schedule.json').write_text(json.dumps(dict(status='MODEL_RW_CHECKED_CONSERVATIVE')))
        r=dict(status='MODELED_NUMERIC_CHECKED',numeric_status='EXECUTED',failed_batches=[],
               structural_reference_passed=True,task_reference_passed=True,schedule_status='MODEL_RW_CHECKED_CONSERVATIVE',
               task=t.frozen(),task_hash=t.identity,precision_policy=policy_for(t.profile).frozen(),
               runtime_contract=HEADER,operator='ns_corrected_v1',algorithm=params,seed=77,asim=receipt)
        def seal():
            r['artifacts']=[dict(path=n,bytes=(root/n).stat().st_size,sha256=sha(root/n)) for n in REQUIRED_ARTIFACTS]
            p=root/'result.json';p.write_text(json.dumps(r));return dict(path=str(p),sha256=sha(p),role='historical_diagnostic')
        return seal(),r,seal,dict(binary=str(binary),runtime_contract=HEADER,operators=['ns_corrected_v1'])

    def test_intake_uses_actual_output_and_recomputed_mmse(self):
        with tempfile.TemporaryDirectory() as d:
            item,r,seal,runtime=self.case(Path(d));loaded=load_case(item,runtime,REPO)
            self.assertEqual(len(loaded['channels']),24)
            self.assertEqual(loaded['algorithm'],r['algorithm'])

    def test_changed_output_and_retagged_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);item,r,seal,runtime=self.case(root)
            (root/'outputs.npy').write_bytes(b'changed')
            with self.assertRaises(Invalid): load_case(item,runtime,REPO)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);item,r,seal,runtime=self.case(root)
            with np.load(root/'channel.npz') as data: values={k:data[k].copy() for k in data.files}
            values['symbols'] *= -1;np.savez(root/'channel.npz',**values);item=seal()
            with self.assertRaisesRegex(Invalid,'fixture mismatch'): load_case(item,runtime,REPO)

    def test_failed_prerequisite_does_not_load_numeric_payload(self):
        with tempfile.TemporaryDirectory() as d:
            item,r,seal,runtime=self.case(Path(d));r['status']='SKIPPED_NOT_COMPETITIVE';item=seal()
            with patch('verification.communication.runner.np.load') as loader:
                with self.assertRaises(Invalid): load_case(item,runtime,REPO)
                loader.assert_not_called()

    def test_calibration_cannot_claim_heldout_quality(self):
        c=contract();c['cases'][0]['role']='calibration'
        with self.assertRaises(Invalid): validate(c)
        with tempfile.TemporaryDirectory() as d:
            item,r,seal,runtime=self.case(Path(d));item['role']='holdout'
            with self.assertRaises(OSError): load_case(item,runtime,REPO)

    def test_duplicate_channel_cannot_inflate_samples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);item,r,seal,runtime=self.case(root)
            c=contract();c['cases']=[item,item];p=root/'runtime.json';p.write_text('{}');c['runtime']=dict(path=str(p),sha256=sha(p))
            import platform
            runtime.update(python_version=platform.python_version(),numpy_version=np.__version__)
            with patch('verification.communication.runner.load_runtime',return_value=runtime):
                with self.assertRaisesRegex(Invalid,'duplicate channel'): evaluate(c,REPO)

    def test_historical_statistics_never_issue_group_quality_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);item,r,seal,runtime=self.case(root)
            c=contract();c['cases']=[item];p=root/'runtime.json';p.write_text('{}');c['runtime']=dict(path=str(p),sha256=sha(p))
            import platform
            runtime.update(python_version=platform.python_version(),numpy_version=np.__version__)
            with patch('verification.communication.runner.load_runtime',return_value=runtime):
                result=evaluate(c,REPO)
            self.assertEqual(result['status'],'DIAGNOSTIC_ONLY')
            self.assertEqual(result['groups'][0]['status'],'DIAGNOSTIC_ONLY')
            self.assertFalse(result['full_communication_certified'])

    def test_cannot_cherry_pick_heldout_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=contract();c['cases'][0]['role']='holdout'
            p=root/'runtime.json';p.write_text('{}');c['runtime']=dict(path=str(p),sha256=sha(p))
            import platform
            runtime=dict(python_version=platform.python_version(),numpy_version=np.__version__)
            g=dict(identity='group',required_holdout_paths=[str(root/'missing/result.json')])
            with patch('verification.communication.runner.load_runtime',return_value=runtime),patch('verification.communication.runner.load_case',return_value=g):
                with self.assertRaisesRegex(Invalid,'omitted heldout'): evaluate(c,REPO)


if __name__=='__main__': unittest.main()
