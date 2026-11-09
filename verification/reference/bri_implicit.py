"""Independent frozen algorithm; never imports actual instruction/replay engine."""
import numpy as np
from verification.contracts.core import Invalid
from verification.studies.fp16_independent import point_solve

def simulate_bri_implicit_fp16(H,Y,regularization,iterations):
    if type(iterations) is not int or not 1<=iterations<=76:raise Invalid('implicit K budget')
    if H.shape[1:]!=(64,16) or Y.shape!=(len(H),64,8):raise Invalid('implicit candidate requires Nr64/Nt16/RHS8')
    b=point_solve(H,Y,regularization,iterations)
    output=b[:,:16,:8].astype(float)+1j*b[:,16:,:8].astype(float)
    return dict(output=output,output_block=b)
