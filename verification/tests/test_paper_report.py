import unittest
from verification.reporting.paper import readiness,require_final,REQUIRED
from verification.contracts.core import Invalid

class PaperReportTests(unittest.TestCase):
    def test_structural_and_precision_pass_not_full_acceptance(self):
        s={'status':'GATES_PASSED','cases':[{'actual':'STRUCTURE_CHECKED','passed':True}]*3}
        p={'status':'GATES_PASSED','cases':[{'status':'MODELED_NUMERIC_CHECKED','bit_equal_batches':96,'passed':True}]*3}
        r=readiness(s,p,{'all_numerics_passed':False})
        self.assertFalse(r['final_eligible']);self.assertEqual(r['gates']['declared_precision_model'],'PASS')
        with self.assertRaises(Invalid):require_final(r)
    def test_missing_required_gate_cannot_finalize(self):
        with self.assertRaises(Invalid):require_final({'gates':{'actual_instruction_replay':'PASS'}})
    def test_unknown_extra_pass_not_a_required_gate(self):
        with self.assertRaises(Invalid):require_final({'gates':{k:'PASS' for k in REQUIRED}|{'self_reported':'PASS'}})
