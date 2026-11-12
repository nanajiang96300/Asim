import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('claude_codegen',Path(__file__).resolve().parents[1]/'scripts/claude_codegen.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class CodegenTests(unittest.TestCase):
    def proposal(self):return {'subtype':'success','is_error':False,'result':json.dumps({'files':[{'path':n,'content':'generated source content\n'*2} for n in sorted(m.ALLOWED)]})}
    def test_missing_source_or_self_score_is_not_acceptance(self):
        for r in [{'subtype':'success','result':'PASS score100'},{'subtype':'error','is_error':True,'result':'{}'}]:
            with self.assertRaises((ValueError,json.JSONDecodeError)):m.files_from_response(r)
    def test_reference_or_traversal_path_rejected(self):
        for path in ['verification/reference/affine.py','../outside.cc']:
            r=self.proposal();d=json.loads(r['result']);d['files'][0]['path']=path;r['result']=json.dumps(d)
            with self.assertRaises(ValueError):m.files_from_response(r)
    def test_duplicate_path_and_json_key_rejected(self):
        r=self.proposal();d=json.loads(r['result']);d['files'][1]=d['files'][0];r['result']=json.dumps(d)
        with self.assertRaises(ValueError):m.files_from_response(r)
        r['result']='{"files":[],"files":[]}'
        with self.assertRaises(ValueError):m.files_from_response(r)
    def test_only_allowed_files_imported_protected_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);subprocess.run(['git','init','-q',str(r)],check=True);(r/'protected.py').write_text('frozen evaluator\n');subprocess.run(['git','add','protected.py'],cwd=r,check=True);subprocess.run(['git','-c','user.email=test@example.invalid','-c','user.name=Test','commit','-qm','fixture'],cwd=r,check=True)
            rows=m.apply_files(m.files_from_response(self.proposal()),r);self.assertEqual(len(rows),3);self.assertEqual((r/'protected.py').read_text(),'frozen evaluator\n')
            with self.assertRaises(ValueError):m.apply_files(m.files_from_response(self.proposal()),r)
