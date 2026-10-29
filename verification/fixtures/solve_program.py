"""Low-level fixture to exercise framework, separate from existing NS operators."""
from verification.shadow.program import Builder
from verification.contracts.core import Contract


def inverse_apply(nt=16,nrhs=8,iterations=12):
    n=2*nt;r=2*nrhs;b=Builder(Contract(contract_id="mmse_ns_inverse_apply_kernel_v1"))
    for name,space,shape in [('A','DRAM',(n,n)),('b','DRAM',(n,r)),('X0','DRAM',(n,n)),('C','DRAM',(n,n)),
                            ('a','SPAD',(n,n)),('rhs','SPAD',(n,r)),('x','SPAD',(n,n)),('c','SPAD',(n,n)),
                            ('t','SPAD',(n,n)),('res','SPAD',(n,n)),('answer','ACCUM',(n,r)),('out','DRAM',(n,r))]:
        b.region(name,space,shape)
    for source,dest in [('A','a'),('b','rhs'),('X0','x'),('C','c')]:b.emit('LOAD',[source],dest)
    for _ in range(iterations):
        b.emit('GEMM',['a','x'],'t').emit('SUB',['c','t'],'res').emit('GEMM',['x','res'],'x')
    return b.emit('GEMM',['x','rhs'],'answer').emit('STORE',['answer'],'out').freeze('out')
