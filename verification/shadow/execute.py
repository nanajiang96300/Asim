"""Structure replay library. Production invocation must follow trusted Asim screening."""
from copy import deepcopy
import numpy as np
from verification.contracts.core import Invalid, digest, screen_asim
from verification.shadow.program import validate
from verification.reconstruction.engine import ArrayMemory,full_view,execute_views

def _replay(program, inputs):
    """Primitive test entry; not the public candidate evaluation path."""
    validate(program)
    memory=ArrayMemory({n:{'rows':r['shape'][0],'cols':r['shape'][1],'name':n} for n,r in program['regions'].items()},np.float64)
    views={n:full_view(n,r['shape']) for n,r in program['regions'].items()}
    for name, value in inputs.items():
        if name not in program['regions'] or program['regions'][name]['space'] != 'DRAM' or name == program['output']:
            raise Invalid('unexpected input: ' + name)
        a = np.asarray(value)
        if np.iscomplexobj(a) or a.dtype != np.float64 or list(a.shape) != program['regions'][name]['shape'] or not np.isfinite(a).all():
            raise Invalid('input shape/dtype/finite mismatch: ' + name)
        memory.write(views[name],a.copy())
    snapshots = []
    stored = set()
    for op in program['ops']:
        code=op['opcode']
        result=execute_views(memory,code,[views[n] for n in op['src']],views[op['dst']])
        if code == 'STORE': stored.add(op['dst'])
        snapshots.append({'op_id': op['id'], 'value': result.tolist()})
    if program['output'] not in stored:
        raise Invalid('missing final STORE')
    return {'status': 'STRUCTURE_EXECUTED', 'program_hash': digest(program), 'output': memory.read(views[program['output']]).tolist(), 'snapshots': snapshots, 'score': None, 'G4': 'NOT_RUN'}

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
