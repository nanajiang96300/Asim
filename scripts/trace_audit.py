#!/usr/bin/env python3
"""
B2 Instruction-Level Trace Audit.
Verifies that SCALAR instructions in trace.csv match FormulaLogger declarations.
Does NOT perform numerical replay — operates at the instruction-count level.
"""
import sys, os, csv, json, re
from collections import defaultdict

TRACE_CATEGORIES = {
    'Cube': 'GEMM',
    'MTE2': 'MOVIN', 
    'MTE3': 'MOVOUT',
    'Vector': 'VECTOR',
    'Scalar': 'SCALAR',
    'Wait': None,  # skip
}


def audit_trace(trace_path, formula_path=None):
    """Audit trace.csv against formula_steps.json."""
    
    # Parse trace — count instructions by type and ID prefix
    trace_ops = defaultdict(lambda: defaultdict(int))  # category → opcode → count
    trace_ids = defaultdict(set)  # instruction ID → set of opcodes
    
    with open(trace_path) as f:
        for row in csv.DictReader(f):
            name = row.get('name', '')
            unit = row.get('unit', '')
            
            # Classify event
            category = None
            for key, cat in TRACE_CATEGORIES.items():
                if key in unit:
                    category = cat
                    break
            if category is None:
                continue
            
            # Extract opcode from name (instruction IDs like "CHOL_NB_POTRF_SQRT_0")
            # Determine opcode from unit
            if 'Cube' in unit:
                opcode = 'GEMM'
            elif 'MTE2' in unit:
                opcode = 'MOVIN'
            elif 'MTE3' in unit:
                opcode = 'MOVOUT'
            elif 'Vector' in unit:
                opcode = 'VECTOR'
            elif 'Scalar' in unit:
                # Infer SCALAR opcode from name pattern
                if 'SQRT' in name:
                    opcode = 'SCALAR_SQRT'
                elif 'DIV' in name or 'DINV' in name or 'UNITY' in name:
                    opcode = 'SCALAR_DIV'
                elif 'SUB' in name:
                    opcode = 'SCALAR_SUB'
                elif 'MUL' in name:
                    opcode = 'SCALAR_MUL'
                elif 'ADD' in name:
                    opcode = 'SCALAR_ADD'
                else:
                    opcode = 'SCALAR_OTHER'
            else:
                opcode = category
            
            trace_ops[category][opcode] += 1
            
            # Extract instruction ID prefix (everything before the last _NUMBER)
            id_prefix = re.sub(r'_\d+$', '', name)
            id_prefix = re.sub(r'_\d+_\d+$', '', id_prefix)
            trace_ids[id_prefix].add(opcode)
    
    # Report trace statistics
    print("=" * 60)
    print("  B2 Trace Audit Report")
    print("=" * 60)
    
    print(f"\n  Trace events: {sum(sum(d.values()) for d in trace_ops.values())}")
    for cat in ['GEMM', 'MOVIN', 'MOVOUT', 'VECTOR', 'SCALAR']:
        if cat in trace_ops:
            total = sum(trace_ops[cat].values())
            print(f"  {cat:<10}: {total:>8}")
            for opcode, count in sorted(trace_ops[cat].items()):
                print(f"    {opcode:<20}: {count:>8}")
    
    # FormulaLogger cross-check
    if formula_path and os.path.exists(formula_path):
        print(f"\n  --- FormulaLogger Cross-Check ---")
        with open(formula_path) as f:
            data = json.load(f)
        
        # Count FormulaLogger steps by op_type
        formula_counts = defaultdict(int)
        formula_relations = defaultdict(set)  # relation_id → op_types
        
        for step in data['steps']:
            formula_counts[step['op_type']] += 1
            formula_relations[step['relation_id']].add(step['op_type'])
        
        # Check: does each FormulaLogger op_type have corresponding trace ops?
        print(f"  FormulaLogger steps: {sum(formula_counts.values())}")
        for op_type, count in sorted(formula_counts.items()):
            print(f"    {op_type:<20}: {count:>8}")
        
        # Instruction count sanity
        n_gemm_trace = trace_ops.get('GEMM', {}).get('GEMM', 0)
        n_gemm_formula = formula_counts.get('GEMM', 0)
        
        n_scalar_trace = sum(trace_ops.get('SCALAR', {}).values())
        n_cholesky_formula = formula_counts.get('CHOLESKY', 0)
        n_trsm_formula = formula_counts.get('TRSM', 0)
        
        print(f"\n  Cross-Check:")
        print(f"    GEMM  — trace: {n_gemm_trace}, formula: {n_gemm_formula}")
        print(f"    SCALAR total: {n_scalar_trace}")
        print(f"    Formula CHOLESKY: {n_cholesky_formula}, TRSM: {n_trsm_formula}")
        
        # Find gaps: FormulaLogger steps without trace coverage
        formula_rels = set(formula_relations.keys())
        trace_id_prefixes = set(trace_ids.keys())
        
        missing_from_trace = formula_rels - trace_id_prefixes
        if missing_from_trace:
            print(f"\n  ⚠️  FormulaLogger steps NOT found in trace:")
            for rid in sorted(missing_from_trace):
                print(f"      relation_id={rid}  op_types={formula_relations[rid]}")
        
        # Find SCALAR instructions without FormulaLogger coverage  
        scalar_ids = {k for k, v in trace_ids.items() if any(o.startswith('SCALAR') for o in v)}
        covered_scalar = scalar_ids & formula_rels
        uncovered_scalar = scalar_ids - formula_rels
        if uncovered_scalar:
            print(f"\n  ⚠️  SCALAR instructions WITHOUT FormulaLogger coverage:")
            for uid in sorted(uncovered_scalar):
                print(f"      {uid}: {trace_ids[uid]}")
        
        return {
            'trace_ops': dict(trace_ops),
            'formula_counts': dict(formula_counts),
            'missing_from_trace': missing_from_trace,
            'uncovered_scalar': uncovered_scalar,
        }
    
    return {'trace_ops': dict(trace_ops)}


if __name__ == '__main__':
    trace_path = sys.argv[1] if len(sys.argv) > 1 else None
    formula_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not trace_path:
        print("Usage: python trace_audit.py <trace.csv> [formula_steps.json]")
        sys.exit(1)
    
    result = audit_trace(trace_path, formula_path)
    
    # Exit code: 0 if no gaps, 1 if gaps found
    gaps = len(result.get('missing_from_trace', set())) + len(result.get('uncovered_scalar', set()))
    sys.exit(0 if gaps == 0 else 1)
