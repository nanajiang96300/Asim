"""Incremental single-operation JSON reader; bounded text buffer, strict EOF.

Instruction objects remain mutable. Repeated exact physical view descriptors
are shared read-only; deepcopy creates independent mutable copies. No opcode
rewriting, missing-input recovery or numerical execution.
"""
import json
import sys
from verification.contracts.core import Invalid


class FrozenView(dict):
    def _readonly(self,*a,**kw):raise TypeError('shared physical view is read-only; deepcopy before editing')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_readonly
    def __deepcopy__(self,memo):
        from copy import deepcopy
        return {deepcopy(k,memo):deepcopy(v,memo) for k,v in self.items()}


def pairs_decoder():
    pool={};keys=frozenset(('base','col','cols','conjugate','row','rows','transpose'))
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise Invalid('duplicate JSON object key')
            result[sys.intern(k)]=v
        if result.keys()==keys and all(type(v) in (int,bool) for v in result.values()):
            token=tuple((k,type(result[k]),result[k]) for k in sorted(keys))
            if token in pool:return pool[token]
            if len(pool)<8192:
                view=FrozenView(result);pool[token]=view;return view
        return result
    return pairs


class Reader:
    def __init__(self,stream,chunk_size):
        self.stream=stream;self.chunk_size=chunk_size;self.buffer='';self.pos=0;self.eof=False
        self.decoder=json.JSONDecoder(object_pairs_hook=pairs_decoder(),parse_constant=lambda x:(_ for _ in ()).throw(Invalid('nonfinite JSON constant')))
    def fill(self):
        self.buffer=self.buffer[self.pos:];self.pos=0
        data=self.stream.read(self.chunk_size);self.buffer+=data;self.eof=not data
    def peek(self):
        while True:
            while self.pos<len(self.buffer) and self.buffer[self.pos].isspace():self.pos+=1
            if self.pos<len(self.buffer):return self.buffer[self.pos]
            if self.eof:return ''
            self.fill()
    def token(self,char):
        if self.peek()!=char:raise Invalid('invalid/truncated operation JSON: expected '+char)
        self.pos+=1
    def value(self):
        self.peek()
        while True:
            try:v,end=self.decoder.raw_decode(self.buffer,self.pos)
            except json.JSONDecodeError as e:
                if self.eof:raise Invalid('invalid/truncated operation JSON') from e
                self.fill();continue
            # A scalar may end at the chunk boundary, e.g. 12 split as 1|2.
            if end==len(self.buffer) and not self.eof:self.fill();continue
            if type(v) in (int,float) and end<len(self.buffer) and self.buffer[end] not in ' \t\r\n,]}':
                if not self.eof:self.fill();continue
            self.pos=end;return v


def load_operation(path,chunk_size=65536):
    if type(chunk_size) is not int or chunk_size<1:raise Invalid('positive JSON chunk size required')
    with open(path) as stream:
        r=Reader(stream,chunk_size);r.token('{');record={}
        if r.peek()!='}':
            while True:
                key=r.value()
                if not isinstance(key,str) or key in record:raise Invalid('invalid/duplicate operation key')
                r.token(':')
                if key=='instructions':
                    r.token('[');items=[]
                    if r.peek()!=']':
                        while True:
                            item=r.value()
                            if not isinstance(item,dict):raise Invalid('instruction must be an object')
                            items.append(item)
                            if r.peek()!=',':break
                            r.token(',')
                    r.token(']');record[key]=items
                else:record[key]=r.value()
                if r.peek()!=',':break
                r.token(',')
        r.token('}')
        if r.peek():raise Invalid('one actual operation required; trailing JSON/data')
        if not isinstance(record.get('instructions'),list):raise Invalid('missing instruction array')
        return record
