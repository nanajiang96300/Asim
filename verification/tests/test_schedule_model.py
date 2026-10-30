import unittest,tempfile
from verification.schedule.hazards import Access,check_schedule,require_runtime_rw
from verification.contracts.core import Invalid,Unsupported

def events(spec):
    return [dict(op=o,kind=k,cycle=c,order=i) for i,(o,k,c) in enumerate(spec)]

class ScheduleModelTests(unittest.TestCase):
    def setUp(self):self.a=Access('SPAD',0,0,0,8);self.b=Access('SPAD',0,0,8,16)
    def test_legal_read_and_write(self):
        ops=[{'id':'w','reads':[],'writes':[self.a]},{'id':'r','reads':[self.a],'writes':[self.b]}]
        r=check_schedule(ops,events([('w','write_visible',1),('r','read_begin',2),('r','read_done',3),('r','write_visible',4)]))
        self.assertEqual(r['status'],'SCHEDULE_MODEL_CHECKED')
    def test_raw_and_waw(self):
        ops=[{'id':'w','reads':[],'writes':[self.a]},{'id':'r','reads':[self.a],'writes':[]},{'id':'later','reads':[],'writes':[self.a]}]
        r=check_schedule(ops,events([('w','write_visible',5),('r','read_begin',2),('r','read_done',3),('later','write_visible',4)]))
        self.assertEqual({f['hazard'] for f in r['faults']},{'RAW','WAW'})
    def test_overwrite_before_read_finishes(self):
        ops=[{'id':'r','reads':[self.a],'writes':[]},{'id':'w','reads':[],'writes':[self.a]}]
        r=check_schedule(ops,events([('r','read_begin',1),('w','write_visible',2),('r','read_done',5)]),[self.a])
        self.assertEqual(r['faults'][0]['hazard'],'WAR')
    def test_partial_initialization(self):
        op={'id':'r','reads':[self.a],'writes':[]};part=Access('SPAD',0,0,0,4)
        r=check_schedule([op],events([('r','read_begin',1),('r','read_done',2)]),[part])
        self.assertEqual(r['faults'][0]['hazard'],'UNINITIALIZED')
    def test_independent_overlap_is_legal(self):
        ops=[{'id':'a','reads':[self.a],'writes':[]},{'id':'b','reads':[self.b],'writes':[]}]
        r=check_schedule(ops,events([('a','read_begin',1),('b','read_begin',2),('b','read_done',4),('a','read_done',5)]),[self.a,self.b])
        self.assertEqual(r['status'],'SCHEDULE_MODEL_CHECKED')
    def test_other_buffer_does_not_alias(self):
        a1=Access('SPAD',0,1,0,8)
        ops=[{'id':'r','reads':[self.a],'writes':[]},{'id':'w','reads':[],'writes':[a1]}]
        r=check_schedule(ops,events([('r','read_begin',1),('w','write_visible',2),('r','read_done',4)]),[self.a])
        self.assertEqual(r['status'],'SCHEDULE_MODEL_CHECKED')
    def test_read_complete_missing_is_unsupported(self):
        with self.assertRaises(Unsupported):check_schedule([{'id':'r','reads':[self.a],'writes':[]}],events([('r','read_begin',1)]),[self.a])
    def test_pipeline_trace_is_not_g4(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Unsupported):require_runtime_rw(d)
    def test_global_dram_identity(self):
        with self.assertRaises(Invalid):Access('DRAM',0,0,0,8).validate()

    def test_partial_overwrite_is_latest_writer(self):
        part=Access('SPAD',0,0,0,4)
        ops=[{'id':'full','reads':[],'writes':[self.a]},{'id':'partial','reads':[],'writes':[part]},{'id':'r','reads':[self.a],'writes':[]}]
        r=check_schedule(ops,events([('full','write_visible',1),('partial','write_visible',5),('r','read_begin',3),('r','read_done',6)]))
        self.assertIn({'hazard':'RAW','writer':'partial','reader':'r'},r['faults'])
    def test_other_core_or_buffer_cannot_initialize(self):
        for other in (Access('SPAD',0,1,0,8),Access('SPAD',1,0,0,8)):
            r=check_schedule([{'id':'r','reads':[self.a],'writes':[]}],events([('r','read_begin',1),('r','read_done',2)]),[other])
            self.assertEqual(r['faults'][0]['hazard'],'UNINITIALIZED')
    def test_existing_unbound_event_file_still_unsupported(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'rw_events.jsonl').write_text('{}\n')
            with self.assertRaises(Unsupported):require_runtime_rw(d)
