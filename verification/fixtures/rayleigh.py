"""Fixed normalized Rayleigh inputs and independent MMSE reference, not a candidate."""
import numpy as np
from verification.contracts.core import Invalid, Unsupported

def channel(seed, nr=4, nt=2, nrhs=1, regularization=0.1, model='rayleigh'):
    if model != 'rayleigh':
        raise Unsupported('CDL-B reserved for a separately validated channel provider')
    if any(type(v) is not int or v <= 0 for v in (nr, nt, nrhs)) or not np.isfinite(regularization) or regularization <= 0:
        raise Invalid('invalid dimensions or regularization')
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((nr,nt)) + 1j*rng.standard_normal((nr,nt))) / np.sqrt(2*nr)
    y = (rng.standard_normal((nr,nrhs)) + 1j*rng.standard_normal((nr,nrhs))) / np.sqrt(2)
    a = h.conj().T @ h + regularization*np.eye(nt)
    b = h.conj().T @ y
    return {'H': h, 'y': y, 'lambda': regularization, 'A': a, 'b': b, 'x_reference': np.linalg.solve(a,b)}


def real_block(z):
    """Explicit real representation used by structure tests, not free complex hardware."""
    z=np.asarray(z)
    return np.block([[z.real,-z.imag],[z.imag,z.real]]).astype(np.float64)


def mmse_batch(seed, batch_size=96, nr=64, nt=16, nrhs=8, snr_db=10.0, iterations=12):
    """Rayleigh/QPSK fixtures; A,b,X0,C preparation is excluded from kernel timing."""
    if type(batch_size) is not int or not 1<=batch_size<=384 or type(iterations) is not int or not 1<=iterations<=32:
        raise Invalid('invalid batch/iteration budget')
    if any(type(n) is not int or n<=0 for n in (nr,nt,nrhs)) or not np.isfinite(snr_db):raise Invalid('invalid channel parameters')
    lam=float(10**(-snr_db/10))
    if not np.isfinite(lam) or lam<=0:raise Invalid('invalid regularization')
    children=np.random.SeedSequence(seed).spawn(batch_size)
    inputs={name:[] for name in ('A','b','X0','C')};task=[];algorithm=[];channels=[]
    for child in children:
        rng=np.random.default_rng(child)
        h=(rng.standard_normal((nr,nt))+1j*rng.standard_normal((nr,nt)))/np.sqrt(2*nr)
        symbols=((2*rng.integers(0,2,(nt,nrhs))-1)+1j*(2*rng.integers(0,2,(nt,nrhs))-1))/np.sqrt(2)
        noise=np.sqrt(lam/2)*(rng.standard_normal((nr,nrhs))+1j*rng.standard_normal((nr,nrhs)))
        y=h@symbols+noise;a=h.conj().T@h+lam*np.eye(nt);rhs=h.conj().T@y
        x0=np.eye(nt,dtype=np.complex128)/np.linalg.norm(a,2)
        inv=x0.copy()
        # Independent complex algorithm, distinct from the real-block dispatcher.
        for _ in range(iterations):inv=inv@(2*np.eye(nt)-a@inv)
        reference=np.linalg.solve(a,rhs);approx=inv@rhs
        for name,value in [('A',a),('b',rhs),('X0',x0),('C',2*np.eye(nt))]:inputs[name].append(real_block(value))
        task.append(real_block(reference));algorithm.append(real_block(approx))
        channels.append({'H':h,'y':y,'symbols':symbols,'A':a,'b':rhs,'x_reference':reference})
    return {'inputs':{k:np.stack(v) for k,v in inputs.items()},'task':np.stack(task),'algorithm':np.stack(algorithm),'channels':channels,
            'metadata':{'seed':seed,'batch_size':batch_size,'Nr':nr,'Nt':nt,'Nrhs':nrhs,'snr_db':snr_db,'lambda':lam,'iterations':iterations,
                        'normalization':'E|H_ij|^2=1/Nr,E|s|^2=1,noise_variance=lambda',
                        'measurement_scope':'given_A_b_X0_C_inverse_and_apply_kernel'}}
