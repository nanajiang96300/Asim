import json,sys,os
from pathlib import Path
REPO=Path('/home/nanajiang/asim-worktrees/tiny-nn-primitives-v1');sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'scripts'))
from verification.tasks.tiny_nn import NNTask,backend
from verification.reconstruction.runner import evaluate
from verification.asim.runner import file_hash
from nn_acceptance import archive_case,check_nonlinear_cost
ROOT=Path(__file__).parent;cfg=json.loads((REPO/'configs/ascend_910b_nn_v1.json').read_text());cfg['dram_config_path']=str(REPO/'configs/ramulator2_configs/HBM2.yaml');cfg['icnt_config_path']=str(REPO/'configs/booksim2_configs/fly_c1_m2.icnt');os.environ['LD_LIBRARY_PATH']=str(REPO/'build/lib')
case=ROOT/'negative-k-v2';binary=REPO/'build/bin/Simulator';r=evaluate('nn_graph_v1',NNTask(),1,92001,binary,cfg,REPO/'build',case,task_backend=backend(NNTask()))
cost=check_nonlinear_cost(case,cfg);first=[]
for b in r.get('batches',[]):
 bad=sorted([s for s in b.get('stage_checks',[]) if not s['bit_equal']],key=lambda s:s['index'] if s['index'] is not None else -1)
 if bad:first.append({'batch':b['batch'],'first_stage':bad[0]['name'],'index':bad[0]['index']})
passed=r['status']=='REPLAY_MISMATCH' and r['asim']['status']=='COMPLETED' and r['schedule_status']=='MODEL_RW_CHECKED_CONSERVATIVE' and bool(r['failed_batches']) and bool(first)
summary={'status':'REAL_BAD_TRANSPOSE_REJECTED' if passed else 'FAILED','actual_status':r['status'],'schedule':r.get('schedule_status'),'cycles':r['asim']['finish_cycle'],'numeric_status':r['numeric_status'],'failed_batches':r.get('failed_batches'),'final_bit_equal_batches':sum(bool(b.get('precision_bit_equal')) for b in r.get('batches',[])),'first_divergences':first,'max_task_error':max(b.get('task_error',0) for b in r.get('batches',[])),'score':r['score'],'binary_sha256':file_hash(binary),'source_mutant_commit':'codex/tiny-nn-negative-k-v2','nonlinear_cost_checks':cost,'archive':archive_case(case)}
(ROOT/'negative-summary-v2.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k not in ['archive','nonlinear_cost_checks','first_divergences','failed_batches']});sys.exit(0 if passed else 1)
