"""Independent FP16 algorithm prototype for Z=half(half(A@B)+C); no replay imports."""
import numpy as np

def simulate_affine(inputs,task):
    a,b,c=(np.asarray(inputs[k],dtype=np.float16) for k in ('A','B','C'));out=[]
    for left,right,bias in zip(a,b,c):
        acc=np.zeros((task.m,task.n),dtype=np.float32)
        for j in range(task.k):
            product=left[:,j:j+1].astype(np.float32)*right[j:j+1,:].astype(np.float32)
            acc=np.asarray(acc+product,dtype=np.float32)
        temporary=np.asarray(acc,dtype=np.float16)
        out.append(np.asarray(temporary.astype(np.float32)+bias.astype(np.float32),dtype=np.float16))
    return {'output':np.stack(out)}
