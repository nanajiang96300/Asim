"""Versioned, deliberately narrow contracts for the first structure prototype."""
from dataclasses import dataclass
import hashlib
import json
import math

class Invalid(ValueError):
    pass

class Unsupported(Invalid):
    pass

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Contract:
    contract_id: str = 'matrix_structure_v1'
    profile: str = 'structure_f64'
    channel: str = 'rayleigh'
    measurement_scope: str = 'given_inputs_kernel'

    def validate(self):
        if self.profile != 'structure_f64':
            raise Unsupported('precision profile not implemented')
        if self.channel != 'rayleigh':
            raise Unsupported('CDL-B extension reserved; no silent Rayleigh fallback')
        if self.measurement_scope != 'given_inputs_kernel':
            raise Unsupported('end-to-end cost not yet implemented')
        if not self.contract_id:
            raise Invalid('missing contract id')

@dataclass(frozen=True)
class ScreenPolicy:
    baseline_cycles: float
    minimum_speedup: float = 1.0
    maximum_speedup: float = 10.0

    def validate(self):
        values = (self.baseline_cycles, self.minimum_speedup, self.maximum_speedup)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
            raise Invalid('finite positive screening values required')
        if self.maximum_speedup < self.minimum_speedup:
            raise Invalid('inverted speedup range')

def screen_asim(receipt, program_hash, policy):
    """Trusted runner calls this with its own receipt, never an Agent self report.

    This is a screening rule, not authentication of receipt provenance.
    The supervised verification.asim.runner authenticates its own subprocess artifacts;
    this standalone rule does not authenticate externally supplied receipts.
    """
    policy.validate()
    if receipt.get('program_hash') != program_hash:
        raise Invalid('program hash mismatch')
    if receipt.get('status') != 'COMPLETED':
        return 'SKIPPED_ASIM_FAILED'
    cycles = receipt.get('finish_cycle')
    if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or not math.isfinite(cycles) or cycles <= 0:
        raise Invalid('invalid Asim cycle count')
    ratio = policy.baseline_cycles / cycles
    if ratio < policy.minimum_speedup:
        return 'SKIPPED_NOT_COMPETITIVE'
    if ratio > policy.maximum_speedup:
        return 'QUARANTINED_IMPLAUSIBLE_SPEEDUP'
    return 'ELIGIBLE_FOR_NUMERICS'


def json_safe(value):
    """Diagnostic serialization only; invalid metadata never becomes valid policy."""
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return repr(value)
