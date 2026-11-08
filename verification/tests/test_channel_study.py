import unittest
import numpy as np
from verification.channels.cdlb import generate,normalize,frequency_response,SPEC
from verification.planning.channel_study import payload,replay_accounting
from verification.contracts.core import Invalid
from verification.contracts.mmse import MMSETask,task_reference
from verification.profiles.numeric import policy_for
from verification.fixtures.rayleigh import real_block

class ChannelStudyTests(unittest.TestCase):
    def test_cir_fourier_sign_delay_and_normalization(self):
        a=np.ones((1,2,3,2),complex);t=np.array([[0.,.25]])
        self.assertTrue(np.allclose(frequency_response(a,t,1.),1-1j))
        H=normalize(frequency_response(a,t,1.));np.testing.assert_allclose(np.sum(abs(H)**2,axis=(1,2)),3)
        with self.assertRaises(Invalid):normalize(np.zeros((1,2,3)))
        with self.assertRaises(Invalid):frequency_response(a,t[:,:1],1.)
    def test_real_cdl_reproducible_and_not_kronecker_surrogate(self):
        a=generate(44001,2);b=generate(44001,2)
        np.testing.assert_array_equal(a['H'],b['H']);self.assertEqual(a['path_coefficients'].shape,(2,64,16,23))
        self.assertLess(float(a['conversion_max_abs_error']),1e-12)
        np.testing.assert_allclose(np.sum(abs(a['H'])**2,axis=(1,2)),16)
        self.assertFalse(np.array_equal(a['H'][0],a['H'][1]))
        with self.assertRaises(Invalid):generate(1,1,{**SPEC,'model':'A'})
    def test_external_channel_replaces_all_actual_inputs_not_only_reference(self):
        t=MMSETask(batch_size=24,nr=8,nt=4,nrhs=2,profile='modeled_fp16_fp32_real_block_v1')
        rng=np.random.default_rng(12);H=normalize(rng.normal(size=(24,8,4))+1j*rng.normal(size=(24,8,4)))
        d=payload(t,H,12,13)
        np.testing.assert_array_equal(d['H'],H)
        np.testing.assert_allclose(d['Y'],H@d['symbols']+d['noise'])
        p=policy_for(t.profile)
        np.testing.assert_array_equal(d['inputs']['H'],p.store(np.stack([real_block(h) for h in H])))
        np.testing.assert_array_equal(d['inputs']['Y'],p.store(np.stack([real_block(y) for y in d['Y']])))
        target=task_reference(d,t);self.assertEqual(target.shape,(24,4,2))
        with self.assertRaises(Invalid):payload(t,H[:1],12,13)

    def test_rejected_frame_slots_are_not_counted_as_completed_batches(self):
        r=replay_accounting(8064,84,29,96)
        self.assertEqual(r['completed_batch_count'],2784)
        self.assertEqual(r['planned_batch_slots'],8064)
        self.assertIsNone(r['partial_frame_completed_batches'])
        self.assertNotIn('batch_attempts',r)
        with self.assertRaises(Invalid):replay_accounting(96,1,2,96)
