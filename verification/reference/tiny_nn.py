"""Independent frozen algorithm; never imports replay/primitive implementations."""
import numpy as np

def simulate(inputs,task,modeled=True):
    dtype=np.float32 if modeled else np.float64
    def st(a):return np.asarray(a,dtype=np.float16 if modeled else np.float64)
    def mm(a,b):
        a=np.asarray(a,dtype=dtype);b=np.asarray(b,dtype=dtype)
        if not modeled:return a@b
        y=np.zeros((a.shape[0],b.shape[1]),dtype)
        for k in range(a.shape[1]):y=np.asarray(y+np.asarray(a[:,k:k+1]*b[k:k+1,:],dtype=dtype),dtype=dtype)
        return st(y)
    def add(a,b):return st(np.asarray(a,dtype=dtype)+np.asarray(b,dtype=dtype))
    def soft(a):
        out=[]
        for row in np.asarray(a,dtype=dtype):
            z=np.asarray(row-np.max(row),dtype=dtype);ex=np.asarray(np.exp(z),dtype=dtype);total=dtype(0)
            for x in ex:total=dtype(total+x)
            out.append(st(ex/total))
        return np.stack(out)
    def gelu(a):
        x=np.asarray(a,dtype=dtype);square=np.asarray(x*x,dtype=dtype);cube=np.asarray(square*x,dtype=dtype)
        t=np.asarray(x+np.asarray(dtype(.044715)*cube,dtype=dtype),dtype=dtype)
        t=np.asarray(dtype(.7978845608028654)*t,dtype=dtype)
        return st(np.asarray(dtype(.5)*x,dtype=dtype)*np.asarray(dtype(1)+np.tanh(t),dtype=dtype))
    outputs=[];evidence={}
    for x in inputs['X']:
        if task.kind in ('softmax','gelu'):y=(soft if task.kind=='softmax' else gelu)(x);stage={'Y':y}
        else:
            if task.kind=='attention':
                q=add(mm(x,inputs['Wq']),inputs['bq']);k=add(mm(x,inputs['Wk']),inputs['bk']);v=add(mm(x,inputs['Wv']),inputs['bv'])
                logits=st(np.asarray(mm(q,k.T),dtype=dtype)*np.asarray(inputs['scale'],dtype=dtype));p=soft(logits)
                u=add(x,add(mm(mm(p,v),inputs['Wo']),inputs['bo']));stage={'Q':q,'K':k,'V':v,'logits':logits,'P':p,'U':u}
            else:u=x;stage={}
            hidden=add(mm(u,inputs['W1']),inputs['b1']);act=gelu(hidden);value=add(mm(act,inputs['W2']),inputs['b2']);y=add(u,value) if task.kind=='attention' else value
            stage.update(hidden=hidden,activation=act,Y=y)
        outputs.append(y)
        for k,v in stage.items():evidence.setdefault(k,[]).append(v)
    return {'output':np.stack(outputs),'evidence':{k:np.stack(v) for k,v in evidence.items()}}
