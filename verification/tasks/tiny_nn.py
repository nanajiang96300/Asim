"""Trusted small-graph tasks: unary units, MLP and genuine attention, same core."""
from dataclasses import dataclass,asdict
import numpy as np
from verification.contracts.core import Invalid,Unsupported,digest
from verification.profiles.numeric import policy_for
from verification.reference.tiny_nn import simulate

@dataclass(frozen=True)
class NNTask:
    kind:str='attention'
    batch_size:int=96
    length:int=32
    hidden:int=32
    ffn:int=64
    profile:str='modeled_fp16_fp32_nn_v1'
    def validate(self):
        if self.kind not in ('softmax','gelu','mlp','attention') or self.batch_size!=96 or type(self.batch_size) is not int:raise Invalid('NN kind/batch')
        if any(type(v) is not int or v<16 or v>128 or v%16 for v in [self.length,self.hidden,self.ffn]) or self.hidden<32:raise Invalid('NN aligned dimensions')
        if self.profile!='modeled_fp16_fp32_nn_v1':raise Unsupported('NN profile')
    @property
    def output_shape(self):return [self.batch_size,self.length,self.hidden]
    def frozen(self):
        self.validate();return {'schema':'tiny-nn-task-v1',**asdict(self),'measurement':'cold_input_weight_dma_to_output_store','softmax':'stable_rows_f32_ascending_sum','gelu':'gelu_tanh_f32_v1','writeback':'FP16_after_every_node','task_tolerance':.01}
    @property
    def identity(self):return digest(self.frozen())

def graph(t):
    t.validate();L,D,F=t.length,t.hidden,t.ffn;inputs=[dict(name='X',shape=[L,D],batched=True)];nodes=[]
    def inp(name,shape):inputs.append(dict(name=name,shape=shape,batched=False))
    def node(name,op,src,shape,**kw):nodes.append(dict(id=name,op=op,src=src,shape=shape,**kw));return name
    def linear(x,w,b,n):
        z=node('mm_'+w,'GEMM',[x,w],[L,n]);return node('add_'+b,'ADD',[z,b],[L,n],broadcast='row_rhs')
    if t.kind in ('softmax','gelu'):out=node('Y',t.kind.upper(),['X'],[L,D])
    else:
        if t.kind=='attention':
            for name in ['q','k','v','o']:inp('W'+name,[D,D]);inp('b'+name,[1,D])
            inp('scale',[L,L]);q=linear('X','Wq','bq',D);k=linear('X','Wk','bk',D);v=linear('X','Wv','bv',D)
            s=node('scores','GEMM',[q,k],[L,L],transpose_rhs=True);s=node('logits','MUL',[s,'scale'],[L,L]);p=node('P','SOFTMAX',[s],[L,L]);v=node('context','GEMM',[p,v],[L,D]);o=linear(v,'Wo','bo',D);u=node('U','ADD',['X',o],[L,D])
        else:u='X'
        inp('W1',[D,F]);inp('b1',[1,F]);inp('W2',[F,D]);inp('b2',[1,D]);h=linear(u,'W1','b1',F);a=node('activation','GELU',[h],[L,F]);v=linear(a,'W2','b2',D);out=node('Y','ADD',[u,v],[L,D]) if t.kind=='attention' else v
    return {'schema':'nn-graph-v1','inputs':inputs,'nodes':nodes,'output':out}

def fixture(t,seed):
    rng=np.random.default_rng(seed);values={}
    for item in graph(t)['inputs']:
        n=item['name'];shape=([t.batch_size] if item['batched'] else [])+item['shape']
        if n=='scale':v=np.full(shape,1/np.sqrt(t.hidden))
        else:v=rng.uniform(-(.5 if n=='X' else .05 if n.startswith('b') else .15),(.5 if n=='X' else .05 if n.startswith('b') else .15),shape)
        values[n]=v
    x=values['X'];x[0]=0;x[1]=1;x[2]=-1
    if t.kind in ('softmax','gelu'):
        x[3]=rng.choice([-8.,-.5,0.,.5,8.],x[3].shape)
        if t.kind=='softmax':x[4]=np.linspace(-1000,1000,t.hidden)
    return {'inputs':{k:v.astype(np.float16) for k,v in values.items()},'math_inputs':values}

def adapter(operator,t,iterations):
    if operator!='nn_graph_v1' or iterations!=1:raise Unsupported('NN operator/iterations')
    g=graph(t);params={'id':operator,'kind':t.kind,'graph':g}
    model={'name':operator,'batch_size':t.batch_size,'graph':g,'algorithm':params,'output_shape':t.output_shape,'task_contract':t.frozen(),'precision_policy':policy_for(t.profile).frozen()}
    return {'mode':'nn_graph_test'},params,{'models':[model]},simulate

def workload(groups,t):
    g=graph(t);expected=[('load.'+i['name'],0) for i in g['inputs']]+[(n['id'],{'GEMM':3,'ADD':10,'MUL':11,'SOFTMAX':8,'GELU':22}[n['op']]) for n in g['nodes']]+[('store',1)]
    for seq in groups.values():
        actual=[(i['id'].split('.',1)[1],i['opcode']) for i in seq if i['opcode']!=25]
        if actual!=expected or sum(i['opcode']==25 for i in seq)!=len(expected)-1:raise Invalid('NN actual graph work/order mismatch')

def captures(t,seq):
    mapping={'add_bq':'Q','add_bk':'K','add_bv':'V','logits':'logits','P':'P','U':'U','add_b1':'hidden','activation':'activation',graph(t)['output']:'Y'}
    if t.kind in ('softmax','gelu'):mapping={'Y':'Y'}
    return {j:{mapping[i['id'].split('.',1)[1]]:i['semantic']['dst']} for j,i in enumerate(seq) if i['id'].split('.',1)[1] in mapping and i['opcode']!=25}

def backend(t):
    return {'captures':lambda seq:captures(t,seq),'adapter':adapter,'fixture':fixture,'task_reference':lambda d,t:simulate(d['math_inputs'],t,False)['output'],
      'dma_shapes':lambda t:[tuple(i['shape']) for i in graph(t)['inputs']]+[(t.length,t.hidden)],
      'algorithm_reference':lambda fn,d,t,i:fn(d['inputs'],t),'blocks':lambda a:a['output'],'decode':lambda v,t,p:v,
      'diagnostics':lambda d:('task_inputs.npz',d['math_inputs']),'workload_check':lambda groups:workload(groups,t)}
