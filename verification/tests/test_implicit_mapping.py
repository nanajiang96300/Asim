"""Operator adapter, signed zero equivalence and bounded streaming regressions."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from verification.contracts.core import Invalid,Unsupported
from verification.contracts.mmse import MMSETask
from verification.operators.catalog import adapter
from verification.profiles.numeric import F16,bit_equal
from verification.schedule.runtime import HEADER,iter_stream,read_stream,check_runtime,FIELDS
from verification.reference.bri_implicit import simulate_bri_implicit_fp16

class ImplicitMapping(unittest.TestCase):
    def test_adapter_frozen_parameters(self):
        t=MMSETask(profile='modeled_fp16_fp32_real_block_v1')
        e,p,m,ref=adapter('bri_implicit_fp16_v1',t,33)
        self.assertEqual(p['momentum'],.9);self.assertEqual(p['block_size_complex'],1)
        self.assertEqual(m['models'][0]['task_contract'],t.frozen())
        self.assertEqual(e['observer_limits']['max_events'],2500000)
        with self.assertRaises(Invalid):adapter('bri_implicit_fp16_v1',t,77)
        with self.assertRaises(Invalid):adapter('bri_implicit_fp16_v1',MMSETask(),33)
    def test_diagonal_lowering_preserves_zeros_and_underflow(self):
        g=np.array([[-0.,0.,-2**-24,2**-24],[1.,-1.,.25,-.25]],np.float16)
        P=np.diag(np.array([.5,2.],np.float16))
        dense=F16.store(F16.calculate('GEMM',[P,g]))
        raw=F16.store(F16.calculate('MUL',[g,np.diag(P)[:,None]]))
        self.assertFalse(bit_equal(raw,dense))
        g0=F16.store(F16.calculate('ADD',[g,np.array([[0.]],np.float16)]))
        lowered=F16.store(F16.calculate('MUL',[g0,np.diag(P)[:,None]]))
        self.assertTrue(bit_equal(lowered,dense))
    def test_independent_zero_rhs(self):
        H=np.zeros((1,64,16),complex);H[0,:16,:]=np.eye(16)
        out=simulate_bri_implicit_fp16(H,np.zeros((1,64,8),complex),.1,3)['output_block']
        self.assertTrue(np.array_equal(out.view(np.uint16),np.zeros(out.shape,np.uint16)))
    def test_streaming_budget_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'events'
            events=[dict(order=i,cycle=i,core=0) for i in range(3)]
            p.write_text('\n'.join(json.dumps(x) for x in [HEADER,*events]))
            self.assertEqual(read_stream(p,3),list(iter_stream(p,3)))
            with self.assertRaises(Unsupported):list(iter_stream(p,3,max_events=2))
            with self.assertRaises(Unsupported):list(iter_stream(p,3,max_bytes=1))
            events[2]['order']=1;p.write_text('\n'.join(json.dumps(x) for x in [HEADER,*events]))
            with self.assertRaises(Invalid):list(iter_stream(p,3))
    def test_event_descriptor_rejected_before_compaction(self):
        inst=dict(id='batch0.op',opcode=10,dest_addr=0x10000000,src_addrs=[0x10000040,0x10000040],size=1,compute_size=4,m=1,k=1,n=1,base_addr=0,src_from_accum=True,operand_id=0,batch=0,semantic={'schema':'source-views-v1'})
        record=dict(header=dict(schema='source-views-program-v1',precision_bytes=2,regions=[]),instructions=[inst])
        dispatch=[dict(batch=0,core=0,spad_buffer=0,accum_buffer=0,cycle=0,ids=[inst['id']])]
        config=dict(num_cores=24,precision=2,dram_req_size=64,core_config={'core_0':{'core_type':'systolic_ws'}})
        for kind,mutation in [('compute_read_done',{'opcode':11}),('compute_read_done',{'src_addrs':[]}),('compute_read_begin',{'semantic':{}})]:
            event={k:inst[k] for k in FIELDS};event.update(order=0,cycle=1,core=0,spad_buffer=0,accum_buffer=0,kind=kind,semantic=inst['semantic']);event.update(mutation)
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'events';p.write_text(json.dumps(HEADER)+'\n'+json.dumps(event)+'\n')
                with self.assertRaisesRegex(Invalid,'instruction/core/buffer binding mismatch|views differ'):
                    check_runtime(record,dispatch,p,Path(td)/'absent.csv',config,5,stream_limits={'max_bytes':100000,'max_events':10})
    def test_campaign_missing_cases_cannot_pass_vacuously(self):
        from verification.reconstruction.implicit_campaign import coverage
        c=dict(cases=[dict(regularization=l,iterations=k) for l,k in [(1.,13),(.1,33),(.01,76)]],channel_seeds=[56001,56002])
        source=dict(status='ASIM_TEMPLATES_CHECKED',cases=[dict(**r,status='MODELED_NUMERIC_CHECKED') for r in c['cases']])
        data=dict(status='DATASET_FROZEN',channels=192,entries=[dict(seed=s) for s in c['channel_seeds']])
        coverage(c,source,data)
        for missing in ([],source['cases'][:2],list(reversed(source['cases']))):
            with self.assertRaises(Invalid):coverage(c,{**source,'cases':missing},data)
        for missing in ([],data['entries'][:1],list(reversed(data['entries']))):
            with self.assertRaises(Invalid):coverage(c,source,{**data,'entries':missing})
if __name__=='__main__':unittest.main()
