"""Read-only observations of actual execution and independent stage targets.

Targets are mathematical expectations, not substitutes for the instruction path.
"""
import numpy as np
from verification.reconstruction.engine import full_view


def capture_plan(header,instructions):
    regions={r['name']:r for r in header['regions']};plan={}
    def add(index,name,region):
        r=regions[region];plan.setdefault(index,{})[name]=full_view(r['base'],(r['rows'],r['cols']))
    for n,i in enumerate(instructions):
        site=(i.get('semantic') or {}).get('site','').split(':')[-1]
        if site in ('CHOL_NB_REG','CHOL_BLK_REG','LDL_NB_REG','LDL_BLK_REG','BRI_REG'):add(n,'regularized_A','aA')
        if i['id'] in ('CHOL_NB_COL_'+str(header['U']-1),'LDL_NB_COL_'+str(header['U']-1)):
            add(n,'factor_L','aL')
            if 'aD' in regions:add(n,'factor_D','aD')
        if site=='BRI_INIT_Y0':
            add(n,'preconditioner','aB');add(n,'initializer','aYk')
        if site=='BRI_Y_UPDATE_' and 'update_first' not in [k for row in plan.values() for k in row]:
            add(n,'update_first','aYnext')
    return plan


def ldl_reference(a):
    """Independent textbook LDL; no engine or source-view imports in arithmetic."""
    n=a.shape[0];l=np.eye(n,dtype=np.complex128);d=np.empty(n,dtype=np.complex128)
    for j in range(n):
        d[j]=a[j,j]-sum(l[j,k]*d[k]*l[j,k].conjugate() for k in range(j))
        for i in range(j+1,n):l[i,j]=(a[i,j]-sum(l[i,k]*d[k]*l[j,k].conjugate() for k in range(j)))/d[j]
    return l,d


def targets(operator,a,regularization,block_size=2):
    result={'regularized_A':a}
    if operator.startswith('cholesky'):result['factor_L']=np.linalg.cholesky(a)
    elif operator.startswith('ldl'):
        l,d=ldl_reference(a);result.update(factor_L=l,factor_D=d[None,:])
    elif operator=='block_richardson':
        n=a.shape[0]
        if block_size!=2 or n%block_size:raise ValueError('stage target supports exact B2 only')
        b=np.zeros_like(a)
        for j in range(0,n,block_size):b[j:j+block_size,j:j+block_size]=np.linalg.solve(a[j:j+block_size,j:j+block_size],np.eye(block_size))
        result.update(preconditioner=b,initializer=np.eye(n))
    return result


def compare(observations,expected):
    rows={}
    for name,target in expected.items():
        obs=observations.get(name)
        if obs is None:rows[name]={'status':'NOT_REACHED'};continue
        if obs['status']!='CAPTURED':rows[name]={k:v for k,v in obs.items() if k!='value'};continue
        value=obs['value']
        if name=='factor_D':value=value[:1,:]
        err=float(np.linalg.norm(value-target)/max(np.linalg.norm(target),1e-15))
        rows[name]={'status':'STAGE_MATCH' if err<=1e-10 else 'STAGE_MISMATCH','relative_error':err,'at_instruction':obs['index']}
        if name=='factor_L':
            err=float(np.linalg.norm(np.tril(value)-target)/max(np.linalg.norm(target),1e-15))
            rows[name]['lower_triangle_error']=err
    return rows
