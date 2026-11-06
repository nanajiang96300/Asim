"""Trusted adapter catalog; separate from seven historical benchmark entries."""
import importlib
import json
from pathlib import Path
from verification.contracts.core import Invalid,Unsupported
from verification.profiles.numeric import policy_for


def adapter(operator,task,iterations):
    entries=json.loads((Path(__file__).resolve().parents[2]/'orchestrator/codex_operators.json').read_text())['operators']
    if operator not in entries:raise Unsupported('unknown operator adapter')
    e=entries[operator]
    if task.scope!=e['scope'] or task.profile not in e.get('profiles',[e.get('profile')]):raise Invalid('adapter/task boundary mismatch')
    if type(iterations) is not int or not 1<=iterations<=32:raise Invalid('iteration budget')
    make_parameters=getattr(importlib.import_module(e['parameter_module']),e['parameter_function'])
    params={'id':operator,**make_parameters(iterations)}
    model={'name':operator,'matrix_m':task.nr,'matrix_k':task.nt,'nrhs':task.nrhs,'batch_size':task.batch_size,'iterations':iterations,'task_contract':task.frozen(),'precision_policy':policy_for(task.profile).frozen()}
    reference=getattr(importlib.import_module(e['reference_module']),e.get('reference_profiles',{}).get(task.profile,e['reference_function']))
    return e,params,{'models':[model]},reference
