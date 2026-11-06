import unittest
import numpy as np
from scripts.archive_checks import indexed,nested
from scripts.audit_solver_stages import require_array
from scripts.check_solver_mapping_archive import validate_stage_coverage

class MappingArchiveTests(unittest.TestCase):
    def test_no_stage_broadcast_or_nonfinite(self):
        for a in (np.ones((1,4,4)),np.full((4,4),np.nan)):
            with self.assertRaises(ValueError):require_array(a,(4,4),'B')
    def test_required_and_safe_unique_artifacts(self):
        row=dict(path='a',bytes=1,sha256='0'*64)
        for entries in ([],[row,row],[dict(row,path='../a')],[dict(row,path='/a')]):
            with self.assertRaises(ValueError):indexed(entries,{'a'})
    def test_nested_digest_must_match(self):
        inner=dict(path='a',bytes=1,sha256='0'*64);outer={'r/a':dict(inner,path='r/a',bytes=2)}
        with self.assertRaises(ValueError):nested(outer,'r',[inner],{'a'})
    def test_stage_duplicate_cannot_substitute_coverage(self):
        names={f'{f}_{c}' for f in ('ldl','bri') for c in ('f64','f16_42','f16_123','shape2')};sizes={n:48 if n.endswith('shape2') else 96 for n in names}
        cases=[dict(case=n,batch_size=sizes[n],passed=True) for n in names];s=dict(status='STAGE_WITNESSES_PASSED',cases=cases);validate_stage_coverage(s,names,sizes)
        cases[-1]=dict(cases[0])
        with self.assertRaises(ValueError):validate_stage_coverage(s,names,sizes)

if __name__=='__main__':unittest.main()
