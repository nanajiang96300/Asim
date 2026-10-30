"""Independent complex Newton–Schulz; no instruction interpreter imports."""
import numpy as np
from verification.contracts.core import Invalid


def simulate_ns(H,Y,regularization,iterations):
    if type(iterations) is not int or not 1<=iterations<=32:raise Invalid('NS iteration budget 1..32')
    B,_,U=H.shape;eye=np.eye(U);outputs=[];initializers=[]
    for h,y in zip(H,Y):
        a=h.conj().T@h+regularization*eye
        # trace(R(A)) is twice trace(A), matching actual real-block scalar sum.
        x=eye/(2*np.trace(a).real);initializers.append(x.copy())
        for _ in range(iterations):x=x@(2*eye-a@x)
        outputs.append(x@(h.conj().T@y))
    return {'output':np.stack(outputs),'X0':np.stack(initializers),'iterations':iterations,'initialization':'I/(2*trace(complex A))'}
