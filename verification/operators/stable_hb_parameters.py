"""Frozen native HB parameters; no numerical execution."""
from verification.contracts.core import Invalid

def parameters(iterations):
    if type(iterations) is not int or not 1<=iterations<=174:raise Invalid('native fixed HB K budget 1..174')
    return dict(iterations=iterations,block_size_complex=1,coefficient_rule='fixed_hb_alpha_15_16',enclosure_rule='scaled_L4_energy_L2_P_trace_9_8',column_energy_domain=[.00390625,8.],energy_sum_domain=[8.,32.],representation='left_complex_columns',initialization='x0=previous=carry=0')
