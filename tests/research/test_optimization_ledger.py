import unittest,tempfile,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from optimization_ledger import append,read,seal
class LedgerTests(unittest.TestCase):
 def test_chain_and_duplicate_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'ledger';append(p,'a','planned',{});append(p,'b','failed',{'score':None})
   self.assertEqual(len(read(p)),2)
   with self.assertRaises(ValueError):append(p,'a','retry',{})
   p.write_text(p.read_text().replace('failed','passed'))
   with self.assertRaises(ValueError):read(p)
 def test_changed_evidence_rejected_before_append(self):
  with tempfile.TemporaryDirectory() as d:
   e=Path(d)/'e';e.write_text('old');s=seal(e);e.write_text('new');p=Path(d)/'ledger'
   with self.assertRaises(ValueError):append(p,'a','run',dict(evidence=[s]))
   self.assertEqual(read(p),[])
