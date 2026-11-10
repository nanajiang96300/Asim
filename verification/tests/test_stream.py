import json,tempfile,unittest
from pathlib import Path
from verification.reconstruction.stream import load_operation
from verification.contracts.core import Invalid

class StreamTests(unittest.TestCase):
    def read(self,text,n):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text(text);return load_operation(p,n)
    def test_chunk_boundaries_and_metadata_order(self):
        value={'execution':None,'header':{'unicode':'中文\\"','number':12345.6},'instructions':[{'id':'a','semantic':{'src':[{'row':0}]}},{'id':'b'}],'tail':True}
        for n in [1,2,7,64,65536]:self.assertEqual(self.read(json.dumps(value,ensure_ascii=False),n),value)
    def test_truncation_and_extra_operations(self):
        text=json.dumps({'instructions':[{'id':'x'}],'header':{}})
        for i in range(len(text)):
            with self.assertRaises(Invalid):self.read(text[:i],3)
        for extra in ['{}','\n{}','x',',']:
            with self.assertRaises(Invalid):self.read(text+extra,3)
    def test_duplicate_and_malformed_rejected(self):
        for text in ['{"instructions":[],"instructions":[]}','{"instructions":[{"x":1,"x":2}]}','{"instructions":[NaN]}','{"instructions":[1]}','{"instructions":[{},]}','{"header":{}}','{"instructions":[],}']:
            with self.assertRaises(Invalid):self.read(text,2)

    def test_shared_views_cannot_change_other_instructions(self):
        from copy import deepcopy
        view=dict(base=1,col=0,cols=1,conjugate=False,row=0,rows=1,transpose=False)
        record=self.read(json.dumps(dict(instructions=[dict(v=view),dict(v=view)])),7)
        a,b=[i['v'] for i in record['instructions']]
        self.assertIs(a,b)
        for mutate in [lambda:a.update(row=4),lambda:a.pop('row'),lambda:a.__setitem__('row',4),lambda:a.clear()]:
            with self.assertRaises(TypeError):mutate()
        copied=deepcopy(record['instructions'][0]);copied['v']['row']=4
        self.assertEqual(a['row'],0)
