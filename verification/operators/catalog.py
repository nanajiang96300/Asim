"""Trusted adapter catalog; separate from seven historical benchmark entries."""
import importlib
import json
from pathlib import Path
from verification.contracts.core import Invalid,Unsupported


def adapter(operator,task,iterations):
    entries=json.loads((Path(__file__).resolve().parents[2]/'orchestrator/codex_operators.json').read_text())['operators']
    if operator not in entries:raise Unsupported('unknown operator adapter')
    e=entries[operator]
    if task.scope!=e['scope'] or task.profile!=e['profile']:raise Invalid('adapter/task boundary mismatch')
    if type(iterations) is not int or not 1<=iterations<=32:raise Invalid('iteration budget')
    params={'id':operator,'iterations':iterations,'initialization':'I/trace(real_block_A)'}
    model={'name':operator,'matrix_m':task.nr,'matrix_k':task.nt,'nrhs':task.nrhs,'batch_size':task.batch_size,'iterations':iterations,'task_contract':task.frozen()}
    reference=getattr(importlib.import_module(e['reference_module']),e['reference_function'])
    return e,params,{'models':[model]},reference
