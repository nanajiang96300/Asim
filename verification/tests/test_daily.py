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
    def sealed(self,d):
        root=Path(d)
        names={'asim.json','schedule.json','outputs.npy','inputs.npz','references.npz','channel.npz','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','config.json','models.json','simulator.log'}
        for n in names:(root/n).write_text(n)
        (root/'asim.json').write_text(json.dumps({'status':'COMPLETED','returncode':0,'binary_sha256':'binary','finish_cycle':10}))
        r={'status':'MODELED_NUMERIC_CHECKED','schedule_status':'MODEL_RW_CHECKED_CONSERVATIVE','numeric_status':'EXECUTED','failed_batches':[],'operator':'ns','algorithm':{'iterations':16},'asim':{'binary_sha256':'binary','finish_cycle':10},'comparison_identity':'task',
           'artifacts':[{'path':n,'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in sorted(names)]}
        (root/'evaluation_result.json').write_text(json.dumps(r));(root/'runtime_pin.json').write_text(json.dumps({'spec_sha256':'spec'}))
        for n in ('evaluation_result.json','runtime_pin.json'):r['artifacts'].append({'path':n,'bytes':(root/n).stat().st_size,'sha256':sha(root/n)})
        p=root/'result.json';p.write_text(json.dumps(r));return p,r
    def test_baseline_artifact_must_be_unchanged(self):
        from verification.reconstruction.daily import baseline
        with tempfile.TemporaryDirectory() as d:
            p,r=self.sealed(d);self.assertEqual(baseline(p,'ns',16,'binary','spec'),(10,'task'))
            (Path(d)/'outputs.npy').write_text('tampered')
            with self.assertRaises(Invalid):baseline(p,'ns',16,'binary','spec')
    def test_baseline_top_level_cannot_override_sealed_cycle(self):
        from verification.reconstruction.daily import baseline
        with tempfile.TemporaryDirectory() as d:
            p,r=self.sealed(d);r['asim']['finish_cycle']=999;p.write_text(json.dumps(r))
            with self.assertRaises(Invalid):baseline(p,'ns',16,'binary','spec')
    def test_baseline_cannot_skip_required_evidence(self):
        from verification.reconstruction.daily import baseline
        with tempfile.TemporaryDirectory() as d:
            p,r=self.sealed(d);r['artifacts']=[];p.write_text(json.dumps(r))
            with self.assertRaises(Invalid):baseline(p,'ns',16,'binary','spec')
    def test_manifest_hash_is_frozen_at_load(self):
        with tempfile.TemporaryDirectory() as d:
            root,p,s=self.make(d);before=sha(p);loaded=load_runtime(p,root)
            p.write_text(p.read_text()+' ')
            self.assertEqual(loaded['_loaded_sha256'],before);self.assertNotEqual(loaded['_loaded_sha256'],sha(p))
