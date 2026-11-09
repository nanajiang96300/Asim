"""Frozen implicit FP16 candidate parameters; no arithmetic."""
from verification.contracts.core import Invalid

def parameters(iterations):
    if type(iterations) is not int or not 1<=iterations<=76:raise Invalid('implicit candidate K budget 1..76')
    return dict(iterations=iterations,block_size_complex=1,momentum=.9,bounds_lambda_factor=.25,bounds_upper=8.,recurrence='implicit_compensated_damped',representation='left_complex_columns',initialization='x0=previous=carry=0')
