import copy
import unittest
from unittest.mock import patch
import numpy as np
from verification.contracts.core import Contract, ScreenPolicy, Invalid, Unsupported, digest, screen_asim
from verification.shadow.program import Builder, validate, work_inventory
from verification.shadow.execute import _replay, execute_after_asim
from verification.fixtures.rayleigh import channel

def program():
    b=Builder()
    for name,space in [('a','DRAM'),('b','DRAM'),('x','SPAD'),('y','SPAD'),('z','ACCUM'),('out','DRAM')]:
        b.region(name,space,(2,2))
    return b.emit('LOAD',['a'],'x').emit('LOAD',['b'],'y').emit('GEMM',['x','y'],'z').emit('STORE',['z'],'out').freeze('out')

class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.p=program()
        self.inputs={'a':np.array([[2.,1.],[0.,3.]]),'b':np.array([[1.,2.],[4.,0.]])}
        # Synthetic receipt tests ONLY the screening rule, not Asim integration.
        self.r={'status':'COMPLETED','program_hash':digest(self.p),'finish_cycle':80}
        self.policy=ScreenPolicy(100,1,4)
    def test_matrix_execution_independent_expected(self):
        result=execute_after_asim(self.p,self.inputs,self.r,self.policy)
        np.testing.assert_array_equal(result['output'],[[6,4],[12,0]])
        self.assertIsNone(result['score']); self.assertEqual(result['G4'],'NOT_RUN')
    def test_screen_rejections_never_execute(self):
        for receipt in [dict(self.r,status='TIMEOUT'),dict(self.r,finish_cycle=101),dict(self.r,finish_cycle=1)]:
            with self.subTest(receipt=receipt), patch('verification.shadow.execute._replay',side_effect=AssertionError('must not run')):
                result=execute_after_asim(self.p,self.inputs,receipt,self.policy)
                self.assertEqual(result['numeric_status'],'NOT_RUN');self.assertIsNone(result['score'])
    def test_invalid_cycles(self):
        for cycles in [0,-1,float('nan'),float('inf'),True,None]:
            with self.subTest(cycles=cycles),self.assertRaises(Invalid):
                screen_asim(dict(self.r,finish_cycle=cycles),digest(self.p),self.policy)
    def test_program_binding(self):
        with self.assertRaises(Invalid):screen_asim(dict(self.r,program_hash='wrong'),digest(self.p),self.policy)
    def test_missing_input(self):
        with self.assertRaisesRegex(Invalid,'UNINITIALIZED'):_replay(self.p,{'a':self.inputs['a']})
    def test_missing_store(self):
        self.p['ops'].pop()
        with self.assertRaisesRegex(Invalid,'STORE'):_replay(self.p,self.inputs)
    def test_unknown_operation(self):
        self.p['ops'][2]['opcode']='INVERSE'
        with self.assertRaises(Unsupported):validate(self.p)
    def test_timing_override_rejected(self):
        self.p['ops'][2]['compute_size']=1
        with self.assertRaises(Invalid):validate(self.p)
    def test_shape_mismatch(self):
        self.p['regions']['z']['shape']=[1,1]
        with self.assertRaises(Invalid):validate(self.p)
    def test_duplicate_id(self):
        self.p['ops'][1]['id']=self.p['ops'][0]['id']
        with self.assertRaises(Invalid):validate(self.p)
    def test_bad_memory_direction(self):
        self.p['ops'][0]['dst']='out'
        with self.assertRaises(Invalid):validate(self.p)
    def test_complex_and_nonfinite_input(self):
        for a in [self.inputs['a'].astype(complex),np.full((2,2),np.nan)]:
            with self.assertRaises(Invalid):_replay(self.p,dict(self.inputs,a=a))
    def test_alias_reads_snapshot(self):
        self.p['ops'][2]['dst']='x';self.p['ops'][3]['src']=['x']
        np.testing.assert_array_equal(_replay(self.p,self.inputs)['output'],[[6,4],[12,0]])
    def test_input_not_modified(self):
        saved=self.inputs['a'].copy();_replay(self.p,self.inputs)
        np.testing.assert_array_equal(saved,self.inputs['a'])
    def test_sub_mutation_changes_result(self):
        self.p['ops'][2]['opcode']='SUB'
        sub=_replay(self.p,self.inputs)['output']
        self.p['ops'][2]['opcode']='ADD'
        add=_replay(self.p,self.inputs)['output']
        self.assertFalse(np.allclose(sub,add))
    def test_work_derived_from_shape(self):
        row=work_inventory(self.p)[2]
        self.assertEqual(row['multiply_count'],8);self.assertEqual(row['reduction_add_count'],4)
    def test_freeze_not_mutable_alias(self):
        b=Builder().region('a','DRAM',(1,1)).region('x','SPAD',(1,1)).region('o','DRAM',(1,1))
        b.emit('LOAD',['a'],'x').emit('STORE',['x'],'o');p=b.freeze('o')
        b.program['ops'].clear();self.assertEqual(len(p['ops']),2)
    def test_policy_bounds(self):
        for p in [ScreenPolicy(0),ScreenPolicy(100,4,1),ScreenPolicy(float('nan'))]:
            with self.assertRaises(Invalid):p.validate()
    def test_rayleigh_reproducible_and_oracle(self):
        f=channel(42);g=channel(42)
        np.testing.assert_array_equal(f['H'],g['H'])
        np.testing.assert_allclose(f['A']@f['x_reference'],f['b'],rtol=1e-12,atol=1e-12)
        self.assertFalse(np.array_equal(f['H'],channel(43)['H']))
    def test_cdlb_no_fallback(self):
        with self.assertRaises(Unsupported):channel(42,model='cdl-b')
        with self.assertRaises(Unsupported):Contract(channel='cdl-b').validate()
    def test_precision_no_silent_fallback(self):
        with self.assertRaises(Unsupported):Contract(profile='fp16').validate()

if __name__=='__main__':unittest.main()
