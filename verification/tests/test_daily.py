import json,tempfile,unittest
from pathlib import Path
from verification.reconstruction.daily import load_runtime,sha
from verification.schedule.runtime import HEADER
from verification.contracts.core import Invalid

class DailyTests(unittest.TestCase):
    def make(self,d):
        root=Path(d)
        for name in ('binary','config','dram'):(root/name).write_text(name)
        spec={'schema':'supervised-reconstruction-runtime-v1','runtime_contract':HEADER,'binary':'binary','config':'config','dependencies':['dram'],'cwd':'.','operators':['ns_corrected_v1'],
              'files':[{'path':n,'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in ('binary','config','dram')]}
        p=root/'pin.json';p.write_text(json.dumps(spec));return root,p,spec
    def test_runtime_dependency_pin_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,s=self.make(d);load_runtime(p,root);(root/'dram').write_text('changed')
            with self.assertRaises(Invalid):load_runtime(p,root)
    def test_runtime_binary_must_be_in_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,s=self.make(d);s['files']=s['files'][1:];p.write_text(json.dumps(s))
            with self.assertRaises(Invalid):load_runtime(p,root)
    def test_unknown_observer_version_not_silent_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,s=self.make(d);s['runtime_contract']={};p.write_text(json.dumps(s))
            with self.assertRaises(Invalid):load_runtime(p,root)

    def test_baseline_must_have_schedule_and_numeric_pass(self):
        from verification.reconstruction.daily import baseline
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';p.write_text(json.dumps({'status':'ALGORITHM_QUALITY_FAILED'}))
            with self.assertRaises(Invalid):baseline(p,'ns',16,'binary')
    def test_baseline_artifact_must_be_unchanged(self):
        from verification.reconstruction.daily import baseline
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'data';f.write_text('proof')
            r={'status':'MODELED_NUMERIC_CHECKED','schedule_status':'MODEL_RW_CHECKED_CONSERVATIVE','numeric_status':'EXECUTED','failed_batches':[],'operator':'ns','algorithm':{'iterations':16},'asim':{'binary_sha256':'binary','finish_cycle':10},'comparison_identity':'task',
               'artifacts':[{'path':'data','bytes':f.stat().st_size,'sha256':sha(f)}]}
            p=Path(d)/'result.json';p.write_text(json.dumps(r));self.assertEqual(baseline(p,'ns',16,'binary'),(10,'task'))
            f.write_text('tampered')
            with self.assertRaises(Invalid):baseline(p,'ns',16,'binary')
