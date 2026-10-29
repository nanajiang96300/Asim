"""Stable comparison metrics: finite inputs do not guarantee finite naive norms."""
import math
import numpy as np
from verification.contracts.core import Invalid


def relative_error(value, reference):
    value=np.asarray(value);reference=np.asarray(reference)
    if value.shape!=reference.shape or not value.size or np.iscomplexobj(value) or np.iscomplexobj(reference) or not np.isfinite(value).all() or not np.isfinite(reference).all():
        raise Invalid('invalid real comparison arrays')
    # Scale BEFORE subtraction/squares; norm(reference) alone may overflow.
    scale=max(1.0,float(np.max(np.abs(value))),float(np.max(np.abs(reference))))
    denominator=max(float(np.linalg.norm(reference/scale)),1e-15/scale)
    numerator=float(np.linalg.norm(value/scale-reference/scale))
    if denominator==0:raise Invalid('unrepresentable error denominator')
    error=numerator/denominator
    if not math.isfinite(error):raise Invalid('non-finite derived relative error')
    return error
