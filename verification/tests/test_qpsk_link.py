import copy
import math
import json
import tempfile
import unittest
import platform
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from verification.communication.qpsk import LABELS,POINTS,modulate,demodulate,nearest,theory,awgn_controls,measurements
from verification.communication.qpsk_link import frame,group_metrics,Replayer,validate,run,sources
from verification.communication.qpsk_sources import validate as validate_sources,guard as guard_sources
from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask
from verification.reconstruction.daily import sha

LIMITS=dict(max_ber_increase=.01,max_ser_increase=.02,max_evm_increase=.01,max_outage_probability=.05,family_alpha=.05,min_independent_channels=192)


class QPSKLinkTests(unittest.TestCase):
    def test_gray_constellation_unit_power_unsigned_bits(self):
        z=modulate(LABELS)
        np.testing.assert_allclose(np.abs(z)**2,1,atol=1e-15)
        np.testing.assert_array_equal(demodulate(z),LABELS)
        np.testing.assert_array_equal(nearest(z),LABELS)
        order=[3,2,0,1,3]
        self.assertTrue(all(np.count_nonzero(LABELS[a]!=LABELS[b])==1 for a,b in zip(order,order[1:])))
        self.assertAlmostEqual(modulate(np.array([0,0],np.uint8)).real,-1/math.sqrt(2))

    def test_axis_ties_and_independent_nearest_slicer(self):
        z=np.array([0.,0j,1j,-1j,1.,-1.,.01-.03j])
        np.testing.assert_array_equal(demodulate(z),nearest(z))
        np.testing.assert_array_equal(demodulate(0j),[1,1])

    def test_modem_rejects_invalid_data(self):
        for b in [np.array([0.,1.]),np.array([0,2]),np.array([],dtype=int),np.array([0,1,0])]:
            with self.assertRaises(Invalid):modulate(b)
        with self.assertRaises(Invalid):demodulate(np.array([np.nan]))

    def test_awgn_es_eb_convention_and_analytic_control(self):
        self.assertAlmostEqual(theory(1.)['ber'],.15865525393145707,places=14)
        self.assertAlmostEqual(theory(.5)['ber'],.07864960352514257,places=14)
        self.assertAlmostEqual(theory(1.)['eb_no_db'],-10*math.log10(2),places=14)
        r=awgn_controls();self.assertEqual(r['status'],'MODEM_CONTROLS_PASSED')
        self.assertEqual(r['noiseless_identity_bit_errors'],0)

    def test_ser_and_confusion_count_single_and_double_bit_errors(self):
        bits=LABELS.reshape(1,1,4,2);truth=modulate(bits);value=truth.copy();value[0,0,0]=value[0,0,0].conjugate()
        r=measurements(value,truth,bits)
        self.assertEqual(r['candidate']['bit_errors'][0],1);self.assertEqual(r['candidate']['symbol_errors'][0],1)
        self.assertEqual(r['candidate_confusion'].sum(),4)
        self.assertEqual(np.trace(r['candidate_confusion']),3)
        value=truth.copy();value[0,0,0]*=-1;r=measurements(value,truth,bits)
        self.assertEqual(r['candidate']['bit_errors'][0],2);self.assertEqual(r['candidate']['symbol_errors'][0],1)

    def test_repeated_frame_keeps_H_and_separates_bits_noise(self):
        t=MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1')
        a=frame(t,22001,33001);b=frame(t,22001,33002)
        np.testing.assert_array_equal(a['H'],b['H']);self.assertFalse(np.array_equal(a['bits'],b['bits']))
        self.assertFalse(np.array_equal(a['noise'],b['noise']))
        np.testing.assert_array_equal(a['Y'],a['H']@modulate(a['bits'])+a['noise'])
        self.assertEqual(set(a['inputs']),{'H','Y','RegI','Identity'})
        self.assertTrue(all(v.dtype==np.float16 for v in a['inputs'].values()))

    def cohort(self,tag,n=96):
        bits=np.tile(LABELS.reshape(1,1,4,2),(n,2,1,1));truth=modulate(bits)
        m=measurements(truth,truth,bits)
        # Two payload frames share the same independent H.
        return dict(channel_seed=tag,channels=n,bits_per_channel=m['bits_per_channel']*2,symbols_per_channel=m['symbols_per_channel']*2,
                    H_hashes=[str(tag)+':'+str(i) for i in range(n)],
                    candidate={k:v*2 for k,v in m['candidate'].items()},mmse={k:v*2 for k,v in m['mmse'].items()},
                    candidate_confusion=m['candidate_confusion']*2,mmse_confusion=m['mmse_confusion']*2)

    def test_frame_aggregation_does_not_inflate_independent_H(self):
        r=group_metrics([self.cohort(1),self.cohort(2)],LIMITS,21)
        self.assertEqual(r['independent_channels'],192)
        self.assertEqual(r['status'],'QPSK_GROUP_CHECKED')
        self.assertEqual(r['candidate_ber'],0);self.assertEqual(r['candidate_ser'],0)
        self.assertGreater(r['outage_bounds']['joint']['upper'],0)
        self.assertLess(r['outage_bounds']['joint']['upper'],.05)
        self.assertEqual(sum(sum(x) for x in r['candidate_confusion']),r['symbols'])

    def test_duplicate_H_and_noisy_bad_frame_cannot_pass(self):
        with self.assertRaises(Invalid):group_metrics([self.cohort(1),self.cohort(1)],LIMITS,21)
        a=self.cohort(1);b=self.cohort(2)
        for v in [a,b]:
            v['candidate']['bit_errors'][:]=v['bits_per_channel']
            v['candidate']['symbol_errors'][:]=v['symbols_per_channel']
        r=group_metrics([a,b],LIMITS,21);self.assertEqual(r['status'],'QUALITY_FAILED')

    def test_source_budget_guards(self):
        c=json.loads(Path('experiments/qpsk_full/2026-11-08-v1/source-contract.json').read_text())
        validate_sources(c);bad=copy.deepcopy(c);bad['budget']['max_new_asim_runs']=1
        with self.assertRaises(Invalid):validate_sources(bad)
        with patch('verification.communication.qpsk_sources.sha',return_value='bad'):
            with self.assertRaisesRegex(Invalid,'producer changed'):guard_sources(c,Path.cwd())

    def test_replay_cannot_pass_only_by_matching_bad_algorithm_reference(self):
        x=object.__new__(Replayer);x.task=MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1')
        x.record={};x.dispatch=[];x.config={};x.r={'algorithm':{}}
        x.reference=lambda *args:dict(output_block=np.zeros((24,8,4)))
        data=frame(x.task,7,8)
        fake=dict(status='STRUCTURE_MATCH',executed=1,total=1,cost_issues=[],_output=np.zeros((8,4)))
        with patch('verification.communication.qpsk_link.validate_actual',return_value=({},dict.fromkeys(range(24),[]),[],{})),patch('verification.communication.qpsk_link.execute_batch',side_effect=lambda *a,**k:copy.deepcopy(fake)):
            r=x.execute(data)
        self.assertTrue(r['precision_bit_equal']);self.assertFalse(r['numeric_passed'])
        self.assertAlmostEqual(max(r['task_errors']),1.)

    def test_incomplete_contract_rejected(self):
        with self.assertRaises(Invalid):validate({})

    def test_source_contract_and_replay_budget_are_preflight_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rp=root/'runtime.json';rp.write_text('{}')
            sc=dict(runtime=dict(path=str(rp),sha256=sha(rp)),source_seed=19001,
                    algorithms=[dict(operator='ns_corrected_v1',iterations=16)],noise_variances=[.1],
                    task=asdict(MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1')))
            (root/'contract.json').write_text(json.dumps(sc));sp=root/'source.json'
            sp.write_text(json.dumps(dict(runtime=sc['runtime'],cases=[dict(operator='ns_corrected_v1',regularization=.1)])))
            repo=Path(__file__).resolve().parents[2]
            c=dict(schema='uncoded-qpsk-link-v1',runtime=sc['runtime'],source_index=dict(path=str(sp),sha256=sha(sp),contract_sha256=sha(root/'contract.json')),
                   evaluator_sources=sources(repo),channel_seeds=[22001,22002],frame_seeds=[33001,33002],limits=dict(LIMITS,min_independent_channels=48),
                   budget=dict(max_batch_replays=1,max_output_bytes=1000000,min_free_bytes=1,max_wall_seconds=100),control_seed=91001,control_symbols=1000)
            rt=dict(python_version=platform.python_version(),numpy_version=np.__version__)
            with patch('verification.communication.qpsk_link.load_runtime',return_value=rt),patch('verification.communication.qpsk_link.load_case') as intake:
                with self.assertRaisesRegex(Invalid,'replay budget'):run(c,repo,root/'output')
                c['source_index']['contract_sha256']='wrong'
                with self.assertRaisesRegex(Invalid,'task contract changed'):run(c,repo,root/'output')
                intake.assert_not_called()


if __name__=='__main__':unittest.main()
