"""Pure projection tests. Actual subprocess gates run via verification.asim.gates."""
import copy
import unittest
from verification.asim.runner import recover
from verification.contracts.core import Invalid, Unsupported
from verification.shadow.program import Builder, validate


def fixture():
    b=Builder().region('a','DRAM',(2,2)).region('x','SPAD',(2,2)).region('o','DRAM',(2,2))
    return b.emit('LOAD',['a'],'x').emit('STORE',['x'],'o').freeze('o')

def exported(p):
    base=dict(size=1,compute_size=0,m=2,k=2,n=2,src_from_accum=False,base_addr=0,batch=0,barrier=False)
    load=dict(base,id='op0',opcode=0,dest_addr=0x10000000,src_addrs=[0x10000],semantic=p['ops'][0])
    barrier=dict(base,id='semantic_barrier_0',opcode=25,dest_addr=0x10000040,src_addrs=[0x10000000],semantic=None,barrier=True,m=0,k=0,n=0,src_from_accum=True)
    store=dict(base,id='op1',opcode=1,dest_addr=0x10000000,src_addrs=[0x10040],semantic=p['ops'][1])
    return {'format':'actual-instructions-v1','header':{k:v for k,v in p.items() if k!='ops'},'instructions':[load,barrier,store]}

class ProjectionTests(unittest.TestCase):
    def test_valid(self):
        p=fixture();self.assertEqual(recover(exported(p),p),p)
    def test_legacy_rejected(self):
        with self.assertRaises(Unsupported):recover({'format':'actual-instructions-v1','header':None},fixture())
    def test_legacy_typed_schema_not_routed_to_generic_adapter(self):
        with self.assertRaises(Unsupported):recover({'format':'actual-instructions-v1','header':{'schema':'legacy-source-views-v1'}},fixture())
    def test_timing_size_mismatch(self):
        p=fixture();r=exported(p);r['instructions'][0]['size']=0
        with self.assertRaises(Invalid):recover(r,p)
    def test_barrier_missing_or_changed(self):
        for change in ('delete','size','source'):
            p=fixture();r=exported(p)
            if change=='delete':r['instructions'].pop(1)
            elif change=='size':r['instructions'][1]['size']=0
            else:r['instructions'][1]['src_addrs']=[123]
            with self.assertRaises(Invalid):recover(r,p)
    def test_semantic_mutation(self):
        p=fixture();r=exported(p);r['instructions'][0]['semantic']=dict(p['ops'][0],dst='o')
        with self.assertRaises(Invalid):recover(r,p)
    def test_memory_total_budget(self):
        p=fixture()
        for j in range(9):p['regions']['extra'+str(j)]={'space':'SPAD','dtype':'float64','shape':[32,32]}
        with self.assertRaises(Unsupported):validate(p)
    def test_reserved_id(self):
        p=fixture();p['ops'][0]['id']='semantic_barrier_0'
        with self.assertRaises(Invalid):validate(p)

class FailureArchiveTests(unittest.TestCase):
    def test_bad_trace_is_archived_before_numerics(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from verification.asim.runner import evaluate
        from verification.contracts.core import ScreenPolicy
        for kind in ('missing','bad_interval','wrong_unit','missing_barrier'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                run=Path(folder)/'run';p=fixture()
                def fake_simulate(binary,config,models,mode,cwd,output):
                    output=Path(output);output.mkdir()
                    (output/'instructions.jsonl').write_text(json.dumps(exported(p))+'\n')
                    if kind!='missing':
                        unit='Core0_Cube' if kind=='wrong_unit' else 'Core0_Vector'
                        end='nan' if kind=='bad_interval' else '2'
                        row='' if kind=='missing_barrier' else f'semantic_barrier_0,{unit},1,{end}\n'
                        (output/'trace.csv').write_text('name,unit,start_cycle,end_cycle\n'+row)
                    return {'status':'COMPLETED','finish_cycle':100}
                with patch('verification.asim.runner.simulate',side_effect=fake_simulate):
                    r=evaluate(p,{'a':np.ones((2,2))},'unused',{'precision':8,'dram_req_size':64},folder,run,ScreenPolicy(100),np.ones((2,2)))
                self.assertEqual(r['status'],'INVALID');self.assertEqual(r['numeric_status'],'NOT_RUN')
                self.assertTrue((run/'result.json').exists());self.assertFalse((run/'numerics_started.json').exists())
