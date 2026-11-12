from pathlib import Path
import json,sys,copy,os,subprocess
REPO=Path('/home/nanajiang/asim-worktrees/tiny-nn-primitives-v1');sys.path.insert(0,str(REPO))
from verification.tasks.tiny_nn import NNTask,adapter,backend
from verification.reconstruction.runner import evaluate
from verification.asim.runner import simulate
ROOT=Path(__file__).parent;binary=REPO/'build/bin/Simulator';cfg=json.loads((REPO/'configs/ascend_910b_nn_v1.json').read_text());cfg['dram_config_path']=str(REPO/'configs/ramulator2_configs/HBM2.yaml');cfg['icnt_config_path']=str(REPO/'configs/booksim2_configs/fly_c1_m2.icnt');os.environ['LD_LIBRARY_PATH']=str(REPO/'build/lib')
rows=[]
for name in ['duplicate_input','unknown_node','bad_output','capacity_overflow','unknown_attribute']:
 t=NNTask(kind='softmax') if name!='capacity_overflow' else NNTask(kind='attention',length=128,hidden=128,ffn=128)
 _,_,models,_=adapter('nn_graph_v1',t,1)
 if name=='duplicate_input':models['models'][0]['graph']['inputs']*=2
 if name=='unknown_node':models['models'][0]['graph']['nodes'][0]['op']='UNKNOWN'
 if name=='unknown_attribute':models['models'][0]['graph']['nodes'][0]['axis']=0
 if name=='bad_output':models['models'][0]['output_shape']=[96,16,32]
 r=simulate(binary,cfg,models,'nn_graph_test',REPO/'build',ROOT/'guards-v2'/name,timeout=30)
 if r['status']=='COMPLETED' or r['returncode']==0:raise AssertionError(name)
 rows.append(dict(case=name,status='BINARY_REJECTED',returncode=r['returncode'],scope='real binary malformed-model preflight, no numerical reference invoked'))
t=NNTask();old=json.loads((REPO/'configs/ascend_910b_quiet.json').read_text());r=evaluate('nn_graph_v1',t,1,92001,binary,old,REPO/'build',ROOT/'guards-v2/old_capacity',task_backend=backend(t));assert r['numeric_status']=='NOT_RUN' and 'asim' not in r and r['status']=='INVALID';rows.append(dict(case='old_capacity_config',status='PREFLIGHT_REJECTED',reason=r['reason']))
out=ROOT/'guards-v2/wrong_profile'
p=subprocess.run([sys.executable,'-m','verification.reconstruction.runner','--task','nn','--profile','structure_f64_real_block','--operator','nn_graph_v1','--binary',str(binary),'--config',str(REPO/'configs/ascend_910b_nn_v1.json'),'--cwd',str(REPO/'build'),'--output',str(out)],cwd=REPO,capture_output=True,text=True);assert p.returncode==2 and not out.exists();rows.append(dict(case='wrong_explicit_profile',status='CLI_REJECTED',returncode=p.returncode))
(ROOT/'guard-summary-v2.json').write_text(json.dumps({'status':'SEVEN_GUARDS_CHECKED','checks':rows},indent=2)+'\n');print(rows)
