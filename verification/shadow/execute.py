"""Structure replay library. Production invocation must follow trusted Asim screening."""
from copy import deepcopy
import numpy as np
from verification.contracts.core import Invalid, digest, screen_asim
from verification.shadow.program import validate

def _replay(program, inputs):
    """Primitive test entry; not the public candidate evaluation path."""
    validate(program)
    memory = {}
    for name, value in inputs.items():
        if name not in program['regions'] or program['regions'][name]['space'] != 'DRAM' or name == program['output']:
            raise Invalid('unexpected input: ' + name)
        a = np.asarray(value)
        if np.iscomplexobj(a) or a.dtype != np.float64 or list(a.shape) != program['regions'][name]['shape'] or not np.isfinite(a).all():
            raise Invalid('input shape/dtype/finite mismatch: ' + name)
        memory[name] = a.copy()
    snapshots = []
    stored = set()
    for op in program['ops']:
        if any(s not in memory for s in op['src']):
            raise Invalid('UNINITIALIZED_READ at ' + op['id'])
        a = [memory[s].copy() for s in op['src']]
        code = op['opcode']
        if code in ('LOAD', 'STORE', 'COPY'): result = a[0]
        elif code == 'ADD': result = a[0] + a[1]
        elif code == 'SUB': result = a[0] - a[1]
        elif code == 'MUL': result = a[0] * a[1]
        elif code == 'GEMM': result = a[0] @ a[1]
        if not np.isfinite(result).all():
            raise Invalid('NONFINITE_RESULT at ' + op['id'])
        memory[op['dst']] = result.copy()
        if code == 'STORE': stored.add(op['dst'])
        snapshots.append({'op_id': op['id'], 'value': result.tolist()})
    if program['output'] not in stored:
        raise Invalid('missing final STORE')
    return {'status': 'STRUCTURE_EXECUTED', 'program_hash': digest(program), 'output': memory[program['output']].tolist(), 'snapshots': snapshots, 'score': None, 'G4': 'NOT_RUN'}

def execute_after_asim(program, inputs, receipt, policy):
    """Internal runner API; receipt provenance must be enforced by that runner.

    No public CLI accepts an Agent-supplied receipt. Production local runs use
    verification.asim.runner, which creates and checks real subprocess artifacts.
    """
    program = deepcopy(program)
    validate(program)
    state = screen_asim(receipt, digest(program), policy)
    if state != 'ELIGIBLE_FOR_NUMERICS':
        return {'status': state, 'numeric_status': 'NOT_RUN', 'score': None}
    return _replay(program, inputs)
