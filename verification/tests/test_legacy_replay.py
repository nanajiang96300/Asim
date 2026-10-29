import copy
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from verification.legacy.replay import Memory,ReplayFault,execute_batch
from verification.reporting.pipeline import statistics


def v(base,row=0,col=0,rows=1,cols=1,transpose=False,conjugate=False):return dict(base=base,row=row,col=col,rows=rows,cols=cols,transpose=transpose,conjugate=conjugate)

def fixture():
    a=0x10000000;r=a+64;g=r+64;total=g+64
    header={'schema':'legacy-source-views-v1','precision_bytes':2,'capacities':[{'spad_bytes':65536,'accum_bytes':65536}],
      'inputs':[{'name':'test.H','base':0x10000,'shape':[1,2,1],'bytes':4},{'name':'test.RegI','base':0x10040,'shape':[1,1],'bytes':2}],
      'outputs':[{'name':'test.A','base':0x10080,'shape':[1,1,1],'bytes':2}],
      'regions':[{'base':a,'rows':2,'cols':1,'name':'H'},*({'base':b,'rows':1,'cols':1,'name':n} for b,n in [(r,'Reg'),(g,'G'),(total,'A')])]}
    def inst(code,src,dst,addrs,dest,operand=0,m=1,k=1,n=1):
        return {'id':str(code),'batch':0,'opcode':code,'src_addrs':addrs,'dest_addr':dest,'operand_id':operand,'m':m,'k':k,'n':n,'barrier':False,
                'semantic':{'schema':'source-views-v1','opcode':code,'src':src,'dst':dst,'site':'unit_fixture'}}
    ops=[inst(0,[v(0x10000,rows=2)],v(a,rows=2),[0x10000],a,100),inst(0,[v(0x10040)],v(r),[0x10040],r,101),
      inst(3,[v(a,rows=2,transpose=True,conjugate=True),v(a,rows=2)],v(g),[a,a],g,m=1,k=2,n=1),
      inst(10,[v(g),v(r)],v(total),[g,r],total),inst(1,[v(total)],v(0x10080),[0x10080],total,200)]
    inputs=[np.array([[[1+1j],[2-1j]]]),np.array([[.1+0j]])]
    return header,ops,inputs

class LegacyTests(unittest.TestCase):
    def test_true_instruction_replay(self):
        h,ops,inputs=fixture();r=execute_batch(h,ops,inputs,0,np.array([[7.1]]))
        self.assertEqual(r['status'],'STRUCTURE_MATCH');self.assertEqual(r['executed'],5)
    def test_actual_add_not_interpreted_as_sub(self):
        h,ops,inputs=fixture();ops[3]['opcode']=18;ops[3]['semantic']['opcode']=18
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'NUMERIC_FAILED')
    def test_partial_initialization_not_full_allocation(self):
        h,ops,inputs=fixture();memory=Memory(h,inputs)
        memory.write(v(0x10000000),np.ones((1,1)))
        with self.assertRaises(ReplayFault):memory.read(v(0x10000000,rows=2))
    def test_source_base_binding(self):
        h,ops,inputs=fixture();ops[2]['semantic']['src'][0]['base']+=64
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'SEMANTIC_BINDING_MISMATCH')
    def test_tensor_bounds(self):
        h,ops,inputs=fixture();ops[0]['semantic']['src'][0]['rows']=100
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'TENSOR_BOUNDS_ERROR')
    def test_explicit_ambiguity_not_guessed(self):
        h,ops,inputs=fixture();ops[2]['semantic']['invalid_reason']='missing column index'
        r=execute_batch(h,ops,inputs,0,np.array([[7.1]]));self.assertEqual(r['status'],'SEMANTIC_AMBIGUITY');self.assertEqual(r['executed'],2)
    def test_opcode_binding(self):
        h,ops,inputs=fixture();ops[2]['opcode']=10
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'SEMANTIC_BINDING_MISMATCH')
    def test_wrong_batch(self):
        h,ops,inputs=fixture()
        self.assertEqual(execute_batch(h,ops,inputs,1,np.array([[7.1]]))['status'],'BATCH_BINDING_MISMATCH')
    def test_missing_load(self):
        h,ops,inputs=fixture();ops.pop(0)
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'UNINITIALIZED_READ')
    def test_pipeline_sums_are_not_makespan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv'
            p.write_text('name,unit,start_cycle,end_cycle\ngemm,Core0_Cube,0,10\nadd,Core0_Vector,5,15\nLoad,Core0_MTE2,2,4\nCubeWait,Core0_Wait,1,5\n')
            instructions=[{'opcode':3,'id':'gemm','barrier':False},{'opcode':10,'id':'add','barrier':False}]
            r=statistics(p,instructions,20,1)
            self.assertEqual(r['status'],'TRACE_VALID');self.assertEqual(r['cores']['0']['busy_union'],15)
            self.assertEqual(r['cores']['0']['idle_in_makespan'],5);self.assertEqual(r['cores']['0']['multi_unit_overlap_union'],7)

    def test_output_shape_must_not_broadcast(self):
        h,ops,inputs=fixture()
        self.assertEqual(execute_batch(h,ops,inputs,0,np.full((2,2),7.1))['status'],'OUTPUT_COVERAGE_UNSUPPORTED')
    def test_multiple_store_is_not_last_store_wins(self):
        h,ops,inputs=fixture();ops.append(copy.deepcopy(ops[-1]))
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'OUTPUT_COVERAGE_UNSUPPORTED')
    def test_duplicate_region_not_silent_alias(self):
        h,ops,inputs=fixture();h['regions'].append(copy.deepcopy(h['regions'][0]))
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'REGION_ALIAS')
    def test_physical_overlap_rejected(self):
        h,ops,inputs=fixture();h['regions'][1]['base']=h['regions'][0]['base']+2
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'REGION_ALIAS')
    def test_configured_capacity_not_allocation_only(self):
        h,ops,inputs=fixture();h['capacities']=[{'spad_bytes':4,'accum_bytes':4}]
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'CAPACITY_ERROR')
    def test_empty_trace_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv';p.write_text('name,unit,start_cycle,end_cycle\n')
            self.assertEqual(statistics(p,[],20,1)['status'],'TRACE_INVALID')
    def test_missing_core_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv';p.write_text('name,unit,start_cycle,end_cycle\ngemm,Core0_Cube,0,10\n')
            self.assertEqual(statistics(p,[],20,2)['status'],'TRACE_INVALID')

    def test_compute_trace_coverage_checks_multiplicity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.csv';p.write_text('name,unit,start_cycle,end_cycle\ngemm,Core0_Cube,0,10\n')
            i={'batch':0,'opcode':3,'id':'gemm','barrier':False};dispatch=[{'batch':0,'core':0}]
            self.assertEqual(statistics(p,[i],20,1,dispatch)['status'],'TRACE_VALID')
            self.assertEqual(statistics(p,[i,i],20,1,dispatch)['status'],'TRACE_INVALID')

    def test_unknown_instruction_schema_rejected(self):
        h,ops,inputs=fixture();ops[0]['semantic']['schema']='future-unknown'
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'SCHEMA_MISMATCH')
    def test_missing_capacity_not_certified(self):
        h,ops,inputs=fixture();del h['capacities']
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'CAPACITY_MISSING')
    def test_load_must_not_cross_batch(self):
        h,ops,inputs=fixture();h['inputs'][0].update(shape=[2,2,1],bytes=8)
        ops[0]['semantic']['src'][0]['base']+=2
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'BATCH_BINDING_MISMATCH')
    def test_offset_store_must_not_cross_task_output(self):
        h,ops,inputs=fixture();h['outputs'][0].update(shape=[1,1,2],bytes=4)
        ops[-1]['semantic']['dst']['base']+=2
        self.assertEqual(execute_batch(h,ops,inputs,0,np.array([[7.1]]))['status'],'OUTPUT_COVERAGE_UNSUPPORTED')
    def test_campaign_requires_all_seven(self):
        from verification.legacy.campaign import require_complete_registry,EXPECTED_ENTRIES
        require_complete_registry(dict.fromkeys(EXPECTED_ENTRIES))
        for registry in ({},{'newton_schulz':{}},dict.fromkeys(EXPECTED_ENTRIES|{'unknown'})):
            with self.assertRaises(ValueError):require_complete_registry(registry)
