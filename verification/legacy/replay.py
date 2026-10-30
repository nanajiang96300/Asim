"""Compatibility imports and historical input fixture; replay is shared."""
import numpy as np
from verification.reconstruction.engine import ReplayFault
from verification.reconstruction.source_views import Memory,execute_batch

def external_inputs(header,seed=42,regularization=.1):
    """Bind actual Model tensors. Never replace Model.H by the A the Op intended."""
    slots=header['inputs'];shape=slots[0]['shape']
    batch,m,u=shape
    rng=np.random.default_rng(seed)
    h=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2*m)
    a=h.conj().transpose(0,2,1)@h+regularization*np.eye(u)
    inverse=np.linalg.solve(a,np.broadcast_to(np.eye(u),a.shape))
    symbols=((2*rng.integers(0,2,(batch,u,u))-1)+1j*(2*rng.integers(0,2,(batch,u,u))-1))/np.sqrt(2)
    y=h@symbols+np.sqrt(regularization/2)*(rng.standard_normal((batch,m,u))+1j*rng.standard_normal((batch,m,u)))
    x0=np.stack([np.eye(u)/np.linalg.norm(x,2) for x in a])
    arrays=[]
    for t in slots:
        name=t['name'].split('.')[-1]
        values={'H':h,'RegI':regularization*np.eye(u),'X_init':x0,'C':2*np.eye(u),'Y':y}
        if name not in values:raise ReplayFault('INPUT_ROLE_UNKNOWN',name)
        value=np.asarray(values[name],dtype=np.complex128)
        if list(value.shape)!=t['shape']:raise ReplayFault('INPUT_SHAPE_MISMATCH',name)
        arrays.append(value)
    return arrays,{'H':h,'A':a,'inverse':inverse,'Y':y,'xhat':np.linalg.solve(a,h.conj().transpose(0,2,1)@y),'lambda':regularization}
