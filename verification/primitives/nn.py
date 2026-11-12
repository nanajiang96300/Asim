"""Frozen host FP32 nonlinear primitives and actual workload validation."""
import numpy as np
from verification.contracts.core import Invalid,Unsupported
CONTRACT='nn-vector-v1'

def validate_instruction(i,profile):
    s=i['semantic'];code=i['opcode']
    if profile!='modeled_fp16_fp32_nn_v1':raise Unsupported('NN primitive requires declared nonlinear precision profile')
    if s.get('kernel_contract')!=CONTRACT:raise Unsupported('unknown NN kernel contract')
    if len(s.get('src',[]))!=1:raise Invalid('NN unary arity')
    a=s['src'][0];d=s['dst'];m,n=d['rows'],d['cols']
    if (a['rows'],a['cols'])!=(m,n) or any(v[k] for v in [a,d] for k in ['row','col','transpose','conjugate']):raise Unsupported('NN full matching unary views required')
    if (i['m'],i['k'],i['n'])!=(m,n,n) or i['compute_size']!=(n*4 if code==8 else m*n*4) or i['size']!=(m*n*2+63)//64:raise Invalid('NN workload mismatch')
    if code==8 and s.get('axis')!=-1:raise Unsupported('softmax axis unsupported')
    if code==22 and s.get('formula')!='gelu_tanh_f32_v1':raise Unsupported('GELU formula unsupported')

def softmax_f32(x):
    z=np.subtract(x,np.max(x,axis=1,keepdims=True),dtype=np.float32)
    e=np.exp(z,dtype=np.float32);total=np.zeros((len(x),1),np.float32)
    for j in range(x.shape[1]):total=np.add(total,e[:,j:j+1],dtype=np.float32)
    return np.divide(e,total,dtype=np.float32)

def gelu_f32(x):
    sq=np.multiply(x,x,dtype=np.float32);cube=np.multiply(sq,x,dtype=np.float32)
    inside=np.add(x,np.multiply(np.float32(.044715),cube,dtype=np.float32),dtype=np.float32)
    arg=np.multiply(np.float32(.7978845608028654),inside,dtype=np.float32)
    return np.multiply(np.multiply(np.float32(.5),x,dtype=np.float32),np.add(np.float32(1),np.tanh(arg,dtype=np.float32),dtype=np.float32),dtype=np.float32)
