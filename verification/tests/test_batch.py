import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from verification.asim.batch import execution_policy, recover_batch, check_events, evaluate_batch
from verification.contracts.core import Invalid, Unsupported, ScreenPolicy
from verification.shadow.metrics import relative_error
from verification.fixtures.rayleigh import mmse_batch
from verification.fixtures.solve_program import inverse_apply
from test_asim_adapter import fixture, exported


def batch_export():
    p=fixture();record=exported(p);record['execution']=execution_policy(24,24)
    original=record['instructions'];record['instructions']=[];dispatch=[]
    for b in range(24):
        ins=copy.deepcopy(original)
        for i in ins:
            i['id']=f"batch{b}."+i['id'];i['batch']=b;i['planned_core']=b
            if i['opcode'] in (0,1):i['src_addrs']=[a+b*128 for a in i['src_addrs']]
        record['instructions'].extend(ins)
        dispatch.append({'batch':b,'core':(b+1)%24,'spad_buffer':1,'accum_buffer':1,'cycle':0,'ids':[i['id'] for i in ins]})
    return p,record,dispatch

class BatchTests(unittest.TestCase):
    def test_valid24_and_dma(self):
        p,r,d=batch_export();a,e,dma,m=recover_batch(r,p,d,24,24)
        self.assertEqual(len(a),24);self.assertEqual(len(e),24);self.assertEqual(sum(dma.values()),48)
        self.assertEqual(set(x['core'] for x in m.values()),set(range(24)))
    def test_bad_batch_stride(self):
        p,r,d=batch_export();r['instructions'][-3]['src_addrs']=[0x10000]
        with self.assertRaises(Invalid):recover_batch(r,p,d,24,24)
    def test_missing_last_batch(self):
        p,r,d=batch_export();r['instructions']=r['instructions'][:-3];d[-1]['ids']=[]
        with self.assertRaises(Invalid):recover_batch(r,p,d,24,24)
    def test_duplicate_dispatch(self):
        p,r,d=batch_export();d[-1]=copy.deepcopy(d[0])
        with self.assertRaises(Invalid):recover_batch(r,p,d,24,24)
    def test_bad_core_or_buffer(self):
        for field,value in [('core',0),('spad_buffer',2),('accum_buffer',True)]:
            p,r,d=batch_export();d[0][field]=value
            with self.assertRaises(Invalid):recover_batch(r,p,d,24,24)
    def test_wrong_global_prefix(self):
        p,r,d=batch_export();r['instructions'][-1]['id']='batch0.op1';d[-1]['ids'][-1]='batch0.op1'
        with self.assertRaises(Invalid):recover_batch(r,p,d,24,24)
    def test_execution_policy(self):
        for b,c in [(0,24),(25,24),(96,2),(True,24),(385,24)]:
            with self.assertRaises(Unsupported):execution_policy(b,c)
    def test_dma_count_gate(self):
        from collections import Counter
        with self.assertRaises(Invalid):check_events([],{},Counter({(0,'Load'):1}),10)
    def test_stable_huge_metrics(self):
        v=np.ones((4,4));r=np.full((4,4),1e308)
        self.assertAlmostEqual(relative_error(v,r),1)
        self.assertEqual(relative_error(r,r),0)
        self.assertEqual(relative_error(np.zeros((2,2)),np.zeros((2,2))),0)
    def test_nonfinite_metrics(self):
        with self.assertRaises(Invalid):relative_error(np.ones((2,2)),np.full((2,2),np.nan))
    def test_invalid_policy_archived(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run'
            r=evaluate_batch(fixture(),{},'unused',{},folder,path,ScreenPolicy(float('nan')),np.zeros((1,2,2)),np.zeros((1,2,2)))
            self.assertEqual(r['status'],'INVALID');self.assertEqual(r['numeric_status'],'NOT_RUN')
            self.assertEqual(json.loads((path/'result.json').read_text())['policy']['baseline_cycles'],'nan')
    def test_independent_batch_fixtures(self):
        f=mmse_batch(42,batch_size=24);g=mmse_batch(42,batch_size=24)
        np.testing.assert_array_equal(f['inputs']['A'],g['inputs']['A'])
        self.assertFalse(np.array_equal(f['inputs']['A'][0],f['inputs']['A'][-1]))
        np.testing.assert_allclose(f['algorithm'],f['task'],rtol=1e-10,atol=1e-12)
        self.assertEqual(f['inputs']['A'].shape,(24,32,32));self.assertEqual(f['task'].shape,(24,32,16))
    def test_cube_alignment(self):
        p=inverse_apply()
        for op in p['ops']:
            if op['opcode']=='GEMM':
                left=p['regions'][op['src'][0]]['shape'];right=p['regions'][op['src'][1]]['shape']
                self.assertTrue(all(d%16==0 for d in [*left,right[1]]))

class BatchRuntimeBoundaryTests(unittest.TestCase):
    def test_huge_algorithm_reference_rejected(self):
        # Unit-only synthetic receipt/projection. The production suite uses real Asim.
        p,record,dispatch=batch_export();a,e,dma,m=recover_batch(record,p,dispatch,24,24)
        events=[]
        for name,unit in e.items():events.append({'name':name,'unit':unit,'start_cycle':1,'end_cycle':2})
        for (core,name),count in dma.items():
            for _ in range(count):events.append({'name':name,'unit':f"Core{core}_MTE{2 if name=='Load' else 3}",'start_cycle':1,'end_cycle':2})
        def fake(binary,config,models,mode,cwd,run_dir,**kwargs):
            import csv
            path=Path(run_dir);path.mkdir()
            (path/'instructions.jsonl').write_text(json.dumps(record)+'\n')
            (path/'dispatch.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in dispatch))
            with (path/'trace.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['name','unit','start_cycle','end_cycle']);writer.writeheader();writer.writerows(events)
            return {'status':'COMPLETED','finish_cycle':100}
        with tempfile.TemporaryDirectory() as folder,patch('verification.asim.batch.simulate',side_effect=fake):
            ref=np.ones((24,2,2));r=evaluate_batch(p,{'a':ref},'unused',{'precision':8,'dram_req_size':64,'scheduler':'simple','num_cores':24},folder,Path(folder)/'run',ScreenPolicy(100),ref,np.full((24,2,2),1e308))
            self.assertEqual(r['status'],'NUMERIC_FAILED');self.assertEqual(r['numeric_status'],'EXECUTED');self.assertEqual(r['failed_batches'],list(range(24)))


class DispatchAndDecisionTests(unittest.TestCase):
    def test_dispatch_temporal_bounds(self):
        from collections import Counter
        d={0:{'cycle':11,'ids':['batch0.semantic_barrier_0']}}
        with self.assertRaises(Invalid):check_events([],{},Counter(),10,d)
        d[0]['cycle']=2
        event={'name':'batch0.semantic_barrier_0','unit':'Core1_Vector','start_cycle':1,'end_cycle':3}
        with self.assertRaises(Invalid):check_events([event],{event['name']:event['unit']},Counter(),10,d)
    def test_equal_ser_counts_do_not_imply_decision_parity(self):
        from verification.asim.batch_gates import diagnostics
        from verification.fixtures.rayleigh import real_block
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);ref=np.array([[1+1j,1+1j]])
            candidate=np.array([[-1+1j,-1+1j]])
            symbols=np.array([[1+1j,-1+1j]])
            np.save(p/'outputs.npy',real_block(candidate)[None,...])
            (p/'trace.csv').write_text('name,unit,start_cycle,end_cycle\n')
            f={'metadata':{'Nt':1,'Nrhs':2},'channels':[{'A':np.eye(1),'b':ref,'symbols':symbols,'x_reference':ref}]}
            r=diagnostics(p,f)
            self.assertEqual(r['candidate_symbol_errors'],r['reference_symbol_errors'])
            self.assertFalse(r['decision_parity']);self.assertEqual(r['decision_mismatches'],2)
