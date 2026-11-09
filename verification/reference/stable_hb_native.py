"""Native adapter for the independent fixed-HB numerical reference."""
from verification.reference.stable_hb import simulate

def reference(H,Y,regularization,iterations):
    r=simulate(H,Y,regularization,iterations);v=r['output_block']
    return dict(output=v[:,:16,:8].astype(float)+1j*v[:,16:,:8].astype(float),output_block=v,evidence=r['setup'])
