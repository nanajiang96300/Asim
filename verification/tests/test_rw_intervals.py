import json,random,tempfile,unittest
from pathlib import Path
from verification.schedule.intervals import Writers,check_schedule_fast
from verification.schedule.hazards import Access,check_schedule
from verification.schedule.runtime import HEADER,read_stream
from verification.contracts.core import Invalid,Unsupported

class IntervalTests(unittest.TestCase):
    def test_interval_assign_matches_independent_bytes(self):
        rng=random.Random(42);index=Writers();bytes_=[None]*32
        for n in range(300):
            lo=rng.randrange(32);hi=rng.randrange(lo+1,33);index.assign(lo,hi,str(n));bytes_[lo:hi]=[str(n)]*(hi-lo)
            values=[None]*32
            for a,b,w in index.query(0,32):values[a:b]=[w]*(b-a)
            self.assertEqual(values,bytes_)
    def test_fast_hazards_match_naive_acceptance(self):
        rng=random.Random(123)
        for attempt in range(100):
            ops=[];events=[];order=0
            for n in range(12):
                lo=rng.randrange(16);hi=rng.randrange(lo+1,17);a=Access('SPAD',rng.randrange(2),0,lo,hi)
                reads=[a] if rng.random()<.7 else [];writes=[a] if not reads or rng.random()<.6 else []
                name=str(n);ops.append(dict(id=name,reads=reads,writes=writes));begin=rng.randrange(20);done=begin+rng.randrange(5)
                for kind,cycle in ([('read_begin',begin),('read_done',done)] if reads else [])+([('write_visible',done+rng.randrange(4))] if writes else []):
                    events.append(dict(op=name,kind=kind,cycle=cycle,order=order));order+=1
            initial=[Access('SPAD',c,0,0,16) for c in range(2)]
            self.assertEqual(check_schedule_fast(ops,events,initial)['status'],check_schedule(ops,events,initial)['status'])
    def test_early_dma_fragment_cannot_hide_behind_last_response(self):
        a=Access('SPAD',0,0,0,128);a0=Access('SPAD',0,0,0,64);a1=Access('SPAD',0,0,64,128)
        ops=[dict(id='read',reads=[a],writes=[]),dict(id='frag0',reads=[],writes=[a0]),dict(id='frag1',reads=[],writes=[a1])]
        es=[dict(op='read',kind='read_begin',cycle=1,order=0),dict(op='read',kind='read_done',cycle=5,order=1),dict(op='frag0',kind='write_visible',cycle=3,order=2),dict(op='frag1',kind='write_visible',cycle=10,order=3)]
        r=check_schedule_fast(ops,es,[a]);self.assertIn({'hazard':'WAR','reader':'read','writer':'frag0'},r['faults'])
    def test_stream_missing_or_malformed_not_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events'
            with self.assertRaises(Unsupported):read_stream(p,10)
            p.write_text(json.dumps(HEADER)+'\n[]\n')
            with self.assertRaises(Invalid):read_stream(p,10)

    def test_epoch_guard_rejects_late_local_effect_not_store_ack(self):
        from verification.schedule.runtime import within_epoch,stamp
        reset0=dict(cycle=0,order=0);reset1=dict(cycle=4,order=20)
        resets={(0,'SPAD'):reset0};times={(0,'SPAD',1):[stamp(reset0),stamp(reset1)]}
        a=Access('SPAD',0,1,0,64);before=dict(cycle=3,order=10);after=dict(cycle=5,order=30)
        self.assertTrue(within_epoch(a,before,0,resets,times))
        self.assertFalse(within_epoch(a,after,0,resets,times))
        self.assertTrue(within_epoch(Access('DRAM',-1,-1,0,64),after,0,resets,times))
    def test_numeric_dma_cannot_apply_free_transpose(self):
        from verification.tests.test_legacy_replay import fixture
        from verification.reconstruction.source_views import execute_batch
        import numpy as np
        h,ops,values=fixture();ops[-1]['semantic']['src'][0]['transpose']=True
        self.assertEqual(execute_batch(h,ops,values,0,np.array([[7.1]]))['status'],'DMA_VIEW_UNSUPPORTED')
