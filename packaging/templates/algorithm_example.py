"""Python-only NS prototype; does not certify a C++ operator or Asim run."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from verification.contracts.mmse import MMSETask,fixture,task_reference
from verification.reference.ns import simulate_ns_fp16_model
from verification.profiles.numeric import policy_for
p=MMSETask(profile='modeled_fp16_fp32_real_block_v1');d=fixture(p,42)
x=simulate_ns_fp16_model(d['H'],d['Y'],p.regularization,16)['output'];ref=task_reference(d,p)
e=np.linalg.norm((x-ref).reshape(p.batch_size,-1),axis=1)/np.linalg.norm(ref.reshape(p.batch_size,-1),axis=1)
print(json.dumps(dict(status='PYTHON_ALGORITHM_ONLY',iterations=16,batch_size=p.batch_size,max_task_relative_error=float(e.max()),task_tolerance=policy_for(p.profile).task_tolerance,operator_verified=False)))
