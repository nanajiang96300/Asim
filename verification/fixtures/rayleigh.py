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
