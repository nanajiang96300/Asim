"""
Channel model interface for MIMO channel matrix generation.

Provides:
- ChannelModel (abstract base)
- RayleighChannel (i.i.d. complex Gaussian)
- CDLBChannel (Kronecker model with 3GPP TR 38.901 CDL-B parameters)
"""

from .rayleigh import RayleighChannel
from .cdl import CDLBChannel

__all__ = ["RayleighChannel", "CDLBChannel"]
