"""Shared supervised entry for Codex and Claude Code. No provider credentials or agent scores."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'agent_profiles/active.json'
ACTIVE=json.loads(PROFILE.read_text()).get('client') if PROFILE.is_file() else None
if ACTIVE not in (None,'codex','claude'):raise ValueError('unknown client profile')
POLICIES=['AGENTS.md','DOCS/research/AGENT_WORKFLOW.md','DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md','.claude/skills/op-flow/SKILL.md']
def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('context');p.add_argument('--client',choices=['codex','claude'],default=ACTIVE);p.add_argument('--task-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    sub.add_parser('doctor');sub.add_parser('evaluate',help='Pass arguments after -- to the unchanged trusted reconstruction runner')
    a,rest=ap.parse_known_args()
    if a.command=='evaluate':
        if rest[:1]==['--']:rest=rest[1:]
        return subprocess.run([sys.executable,'-m','verification.reconstruction.runner',*rest],cwd=ROOT).returncode
    if rest:ap.error('unrecognized arguments: '+' '.join(rest))
    if a.command=='doctor':
        checks={}
        for name in ['codex','claude']:
            path=shutil.which(name);checks[name]={'available':bool(path)}
            if path:
                v=subprocess.run([path,'--version'],capture_output=True,text=True,timeout=10);checks[name]['version']=v.stdout.strip() if v.returncode==0 else 'version command failed'
        print(json.dumps({'active_client':ACTIVE,'clients':checks,'python':sys.version.split()[0],'provider_api_tested':False,'workflow':'supervised; shared evaluator'},ensure_ascii=False));return 0
    if a.client is None:ap.error('context requires --client on the shared branch')
    task=a.task_file.read_text();entries=[]
    for name in POLICIES:
        p=ROOT/name;entries.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    doc={'schema':'asim-agent-task-v1','client':a.client,'base_commit':commit,'task':task,'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'policies':entries,'constraints':{'evaluator_reference_threshold_changes_allowed':False,'build_jobs':1,'order':['freeze','asim','events_and_performance_filter','actual_replay','same_algorithm_reference','fp64_task_reference','applicable_communication','score_and_archive'],'score_before_acceptance':None,'unattended_controller':False}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(doc,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'TASK_PREPARED','path':str(a.output),'base_commit':commit,'agent_executed':False},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
