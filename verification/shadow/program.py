"""One canonical low-level structure program; no candidate timing overrides."""
from copy import deepcopy
from verification.contracts.core import Contract, Invalid, Unsupported, digest

SPACES = {'DRAM', 'SPAD', 'ACCUM'}
ARITY = {'LOAD': 1, 'STORE': 1, 'COPY': 1, 'ADD': 2, 'SUB': 2, 'MUL': 2, 'GEMM': 2}

def positive_int(x):
    return type(x) is int and x > 0

def validate(program):
    if set(program) != {'schema', 'contract', 'regions', 'ops', 'output'} or program['schema'] != 'structure-program-v1':
        raise Invalid('invalid program schema')
    c = Contract(**program['contract']); c.validate()
    regions = program['regions']
    if not regions or not isinstance(regions, dict):
        raise Invalid('regions required')
    for name, r in regions.items():
        if set(r) != {'space', 'shape', 'dtype'} or r['space'] not in SPACES:
            raise Invalid('invalid region: ' + name)
        if r['dtype'] != 'float64':
            raise Unsupported('only real float64 structure mode supported')
        if len(r['shape']) != 2 or not all(positive_int(x) for x in r['shape']):
            raise Invalid('positive two dimensional shape required')
    ids = set()
    for op in program['ops']:
        if set(op) != {'id', 'opcode', 'src', 'dst'}:
            raise Invalid('unknown fields: timing, slicing and immediates cannot be guessed')
        code = op['opcode']
        if code not in ARITY:
            raise Unsupported('unsupported opcode: ' + str(code))
        if not isinstance(op['id'], str) or not op['id'] or op['id'] in ids:
            raise Invalid('duplicate or missing op id')
        ids.add(op['id'])
        if len(op['src']) != ARITY[code] or any(s not in regions for s in op['src']) or op['dst'] not in regions:
            raise Invalid('invalid operand binding')
        src = [regions[s] for s in op['src']]; dst = regions[op['dst']]
        shapes = [s['shape'] for s in src]
        if code == 'GEMM':
            if shapes[0][1] != shapes[1][0] or dst['shape'] != [shapes[0][0], shapes[1][1]]:
                raise Invalid('GEMM shape mismatch')
        elif any(s != dst['shape'] for s in shapes):
            raise Invalid('elementwise/copy shape mismatch')
        if code == 'LOAD':
            if src[0]['space'] != 'DRAM' or dst['space'] == 'DRAM':
                raise Invalid('LOAD direction')
        elif code == 'STORE':
            if src[0]['space'] == 'DRAM' or dst['space'] != 'DRAM':
                raise Invalid('STORE direction')
        elif dst['space'] == 'DRAM' or any(s['space'] == 'DRAM' for s in src):
            raise Invalid('arithmetic must operate on local memory')
    if program['output'] not in regions or regions[program['output']]['space'] != 'DRAM':
        raise Invalid('output must be DRAM')
    return program

class Builder:
    def __init__(self, contract=None):
        self.program = {'schema': 'structure-program-v1', 'contract': vars(contract or Contract()), 'regions': {}, 'ops': [], 'output': None}

    def region(self, name, space, shape):
        if name in self.program['regions']:
            raise Invalid('duplicate region')
        self.program['regions'][name] = {'space': space, 'shape': list(shape), 'dtype': 'float64'}
        return self

    def emit(self, opcode, src, dst):
        self.program['ops'].append({'id': 'op' + str(len(self.program['ops'])), 'opcode': opcode, 'src': list(src), 'dst': dst})
        return self

    def freeze(self, output):
        result = deepcopy(self.program); result['output'] = output
        validate(result)
        return result

def work_inventory(program):
    """Logical work only, NOT an Asim cost model or cycle estimator."""
    validate(program)
    rows = []
    for op in program['ops']:
        m, n = program['regions'][op['dst']]['shape']
        row = {'op_id': op['id'], 'opcode': op['opcode'], 'output_elements': m*n}
        if op['opcode'] == 'GEMM':
            k = program['regions'][op['src'][0]]['shape'][1]
            row.update(m=m, n=n, k=k, multiply_count=m*n*k, reduction_add_count=m*n*max(k-1, 0))
        if op['opcode'] in ('LOAD', 'STORE', 'COPY'):
            row['logical_bytes'] = m*n*8
        rows.append(row)
    return rows
