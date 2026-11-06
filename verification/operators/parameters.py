"""Trusted algorithm parameter adapters; no numerical execution."""
from verification.contracts.core import Invalid

def ns_parameters(iterations):
    return {'iterations':iterations,'initialization':'I/trace(real_block_A)'}

def cholesky_parameters(iterations):
    if iterations!=1:raise Invalid('direct Cholesky accepts iterations=1 sentinel only; it has no iteration parameter')
    return {'factorization':'right_looking_rank1','initialization':'L=0;rhs=H^T*Y'}
