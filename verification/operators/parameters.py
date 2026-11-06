"""Trusted algorithm parameter adapters; no numerical execution."""
from verification.contracts.core import Invalid

def ns_parameters(iterations):
    return {'iterations':iterations,'initialization':'I/trace(real_block_A)'}

def cholesky_parameters(iterations):
    if iterations!=1:raise Invalid('direct Cholesky accepts iterations=1 sentinel only; it has no iteration parameter')
    return {'factorization':'right_looking_rank1','initialization':'L=0;rhs=H^T*Y'}

def ldl_parameters(iterations):
    if iterations!=1:raise Invalid('direct LDL requires iterations=1 sentinel')
    return {'factorization':'right_looking_ldl','initialization':'L=I;D=diag(A_schur);row_weighted_solve'}

def bri_parameters(iterations):
    if type(iterations) is not int or not 1<=iterations<=256:raise Invalid('BRI iteration budget 1..256')
    return {'iterations':iterations,'block_size_real':2,'omega':.5,'recurrence':'precomputed_stationary','initialization':'x0=0'}
