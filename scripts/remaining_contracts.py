"""Frozen expected case and evidence contracts for the 2026-11-08 batch."""
import copy

FAMILIES=('cholesky_block','ldl_block','cholesky_merge')
PARAMS={
'ns_corrected_v1':dict(iterations=16,initialization='I/trace(real_block_A)'),
'cholesky_explicit_v1':dict(factorization='right_looking_rank1',initialization='L=0;rhs=H^T*Y'),
'ldl_explicit_v1':dict(factorization='right_looking_ldl',initialization='L=I;D=diag(A_schur);row_weighted_solve'),
'bri_explicit_v1':dict(iterations=192,block_size_real=2,omega=.5,recurrence='precomputed_stationary',initialization='x0=0'),
'cholesky_block_explicit_v1':dict(factorization='blocked_right_looking_cholesky',block_size_real=4,initialization='L=0;row_triangular_solve'),
'ldl_block_explicit_v1':dict(factorization='blocked_right_looking_ldl',block_size_real=4,initialization='L=I;D;row_weighted_solve'),
'cholesky_merge_explicit_v1':dict(factorization='left_looking_gemm_reduction',block_size_real=1,initialization='L=0;gemm_triangular_solve')}

def case_contract(name):
    if name.endswith('_regression'):
        op=name[:-11]
        if op not in ('ns_corrected_v1','cholesky_explicit_v1','ldl_explicit_v1','bri_explicit_v1'):raise ValueError('unknown regression')
        profile='modeled_fp16_fp32_real_block_v1';seed=42;shape=False
    else:
        found=[f for f in FAMILIES if name.startswith(f+'_')]
        if len(found)!=1:raise ValueError('unknown case')
        family=found[0];kind=name[len(family)+1:]
        if kind not in ('f64','f16_42','f16_123','shape2'):raise ValueError('unknown case suffix')
        op=family+'_explicit_v1';profile='modeled_fp16_fp32_real_block_v1' if kind.startswith('f16') else 'structure_f64_real_block';seed=123 if kind=='f16_123' else 42;shape=kind=='shape2'
    task=dict(batch_size=48 if shape else 96,nr=32 if shape else 64,nt=8 if shape else 16,nrhs=4 if shape else 8,regularization=.1,channel='rayleigh',scope='mmse_h_y_to_xhat_v1',profile=profile)
    return dict(operator=op,precision=profile,seed=seed,algorithm={'id':op,**PARAMS[op]},task=task,status='MODELED_NUMERIC_CHECKED' if profile.startswith('modeled') else 'STRUCTURE_CHECKED')

def validate_case(name,r):
    spec=case_contract(name)
    if any(r.get(k)!=spec[k] for k in ('operator','precision','seed','algorithm','status')) or any(r['task'].get(k)!=v for k,v in spec['task'].items()):raise ValueError('case identity/task/profile/parameter mismatch')
    return spec

def validate_checks(name,checks):
    required={'L','D','unit_diagonal','positive_D'} if name.startswith('ldl_block_') else {'L'}
    if set(checks)!=required or any(type(v) is not bool or not v for v in checks.values()):raise ValueError('missing or failed factor witness role')

def normalize_probe(record):
    normalized=copy.deepcopy(record);header=normalized['header'];header.pop('declared_compute_size',None)
    matches=[i for i in normalized['instructions'] if i['id']=='probe.scalar']
    if len(matches)!=1:raise ValueError('probe scalar missing')
    matches[0]['compute_size']=1
    return normalized
