"""Bounded Claude Code source proposals; API configuration stays in process memory.

This worker proposes files only. It cannot select the evaluator or declare acceptance.
"""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'DOCS/operators/AFFINE_GEMM_DEMO_V1.md','src/inverse/affine_gemm_v1/AffineGemmModel.h','src/inverse/affine_gemm_v1/AffineGemmModel.cc'}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d
def files_from_response(response):
    if response.get('is_error') or response.get('subtype')!='success':raise ValueError('Claude did not complete successfully')
    text=response['result'].strip()
    if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
    doc=json.loads(text,object_pairs_hook=unique)
    if not isinstance(doc,dict) or set(doc)!={'files'} or not isinstance(doc['files'],list) or len(doc['files'])!=3:raise ValueError('three-file source proposal required; a plan or self-score is not code')
    names=[]
    for e in doc['files']:
        if not isinstance(e,dict) or set(e)!={'path','content'} or e['path'] not in ALLOWED or not isinstance(e['content'],str) or not 20<len(e['content'].encode())<131072:raise ValueError('source proposal path/content rejected')
        names.append(e['path'])
    if set(names)!=ALLOWED or len(set(names))!=3:raise ValueError('duplicate or missing candidate path')
    return doc['files']
def apply_files(files,target,overwrite=False):
    target=Path(target).resolve()
    if target==ROOT:raise ValueError('use an independent candidate workspace')
    actual=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=target,text=True).strip()).resolve()
    if actual!=target or subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=target,text=True).strip():raise ValueError('candidate Git workspace must be clean')
    protected={}
    for name in subprocess.check_output(['git','ls-files','-z'],cwd=target).decode().split('\0'):
        if name and name not in ALLOWED and (target/name).is_file():protected[name]=sha(target/name)
    for e in files:
        p=target/e['path']
        if not p.resolve().is_relative_to(target) or p.is_symlink() or (p.exists() and not overwrite):raise ValueError('candidate path exists/escapes; explicit replacement required')
    rows=[]
    for e in files:
        p=target/e['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(e['content'],encoding='utf-8');rows.append({'path':e['path'],'bytes':p.stat().st_size,'sha256':sha(p)})
    if any(not (target/n).is_file() or sha(target/n)!=h for n,h in protected.items()):raise ValueError('protected source changed during import')
    return rows
def configured_env():
    env=os.environ.copy();p=Path.home()/'.claude/settings.json'
    if p.is_file():
        for k,v in json.loads(p.read_text()).get('env',{}).items():
            if k.startswith(('ANTHROPIC_','CLAUDE_CODE_')) and k not in env:env[k]=str(v)
    key=env.get('ANTHROPIC_API_KEY') or env.get('ANTHROPIC_AUTH_TOKEN')
    if not key:raise ValueError('configure the user DeepSeek API key; do not put it in this repository')
    env['ANTHROPIC_API_KEY']=key;env.pop('ANTHROPIC_AUTH_TOKEN',None);env['MAX_THINKING_TOKENS']='0';env['CLAUDE_CODE_MAX_OUTPUT_TOKENS']='32768';env['CLAUDE_CODE_EFFORT_LEVEL']='low'
    return env
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--task-file',type=Path);ap.add_argument('--response',type=Path,help='offline validation/import of a preserved CLI response; no API call');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--apply-root',type=Path);ap.add_argument('--replace-candidate-files',action='store_true');ap.add_argument('--timeout',type=int,default=600);a=ap.parse_args()
    if bool(a.task_file)==bool(a.response):ap.error('choose task-file or response')
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();receipt={'schema':'claude-source-proposal-v1','agent_executed':False,'operator_verified':False,'score':None,'reasoning_time_seconds':None}
    try:
        if a.response:raw=a.response.read_text()
        else:
            prompt=a.task_file.read_text();(a.output/'task.txt').write_text(prompt);env=configured_env();cmd=['claude','--bare','--print','--tools','','--no-session-persistence','--output-format','json','--strict-mcp-config','--effort','low','--settings','{"alwaysThinkingEnabled":false}','--system-prompt','Generate complete source files from the frozen Python algorithm and interface. Return only the requested JSON files bundle. No tools, no questions, no success claims.']
            p=subprocess.run(cmd,cwd=a.apply_root or ROOT,env=env,input=prompt,capture_output=True,text=True,timeout=a.timeout);raw=p.stdout;receipt.update(agent_executed=True,cli_returncode=p.returncode,bare=True,tools_requested=[],max_output_tokens_requested=32768,max_thinking_tokens_requested=0)
            if p.returncode:raise ValueError('Claude process failed; no candidate imported')
        (a.output/'response.json').write_text(raw);response=json.loads(raw);files=files_from_response(response)
        receipt.update(status='SOURCE_PROPOSED',response_sha256=sha(a.output/'response.json'),usage_reported_by_cli=response.get('usage'),model_usage_reported_by_cli=response.get('modelUsage'))
        if a.apply_root:receipt['files']=apply_files(files,a.apply_root,a.replace_candidate_files);receipt['status']='SOURCE_IMPORTED'
    except (ValueError,KeyError,OSError,subprocess.SubprocessError,json.JSONDecodeError) as e:receipt.update(status='REJECTED',reason=str(e))
    receipt['wall_seconds']=time.perf_counter()-start;(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ('usage_reported_by_cli','model_usage_reported_by_cli','files')}));return 1 if receipt['status']=='REJECTED' else 0
if __name__=='__main__':raise SystemExit(main())
