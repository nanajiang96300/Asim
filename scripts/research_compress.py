"""Losslessly retire large old trace copies with a checked restoration ledger."""
import argparse,gzip,hashlib,json,os,shutil,time,uuid
from pathlib import Path

ROOTS=['results/runtime_rw/2026-10-30-v3-frozen',
 'results/legacy_semantics/2026-10-30-seven-b96-raw',
 'results/reconstruction_campaigns/2026-10-29-legacy7-b96-v1',
 'results/generality/2026-11-06-cholesky-v1',
 'results/operator_mapping/2026-11-06-ldl-bri-v1',
 'results/remaining_mapping/2026-11-08-block-merge-v1']
NAMES={'rw_events.jsonl','instructions.jsonl','trace.csv','formula.json','pipeline_statistics.json','instruction_pipeline_detail.csv'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(p,v):
    tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def relative(repo,p):
    p=p.resolve();p.relative_to(repo)
    return str(p.relative_to(repo))

def plan(repo,path):
    if path.exists():raise ValueError('refusing existing plan')
    protected=set()
    spec=json.loads((repo/'orchestrator/codex_runtime.json').read_text())
    for e in spec['files']:protected.add((repo/e['path']).resolve())
    import subprocess
    tracked={repo/p for p in subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()}
    entries=[]
    for root in ROOTS:
        for p in (repo/root).rglob('*'):
            if p.name not in NAMES or not p.is_file() or p.is_symlink() or p.stat().st_size<8*1024**2 or p.stat().st_nlink!=1 or p in tracked:continue
            rel=relative(repo,p)
            if p.resolve() in protected or rel.startswith('results/bri_implicit/') or rel.startswith('results/qpsk_full/'):raise ValueError('protected artifact selected')
            if p.with_name(p.name+'.gz').exists():raise ValueError('compressed destination exists')
            entries.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),archive=rel+'.gz',reason='older diagnostic trace; exact bytes retained in verified gzip; inputs/results/logs/binaries retained'))
    entries.sort(key=lambda e:e['bytes'],reverse=True)
    result=dict(schema='research-lossless-retirement-v1',source_script_sha256=sha(__file__),protected_runtime_sha256=sha(repo/'orchestrator/codex_runtime.json'),roots=ROOTS,entries=entries,authorisation='User 2026-11-09 permits removing unneeded data while retaining latest results, key intermediate data and conclusions',target_free_bytes=6*1024**3,max_files=64,max_wall_seconds=600)
    path.parent.mkdir(parents=True,exist_ok=True);save(path,result);print(json.dumps(dict(plan=str(path),sha256=sha(path),files=len(entries),bytes=sum(e['bytes'] for e in entries))))

def execute(repo,path,approved,ledger):
    if sha(path)!=approved:raise ValueError('plan SHA mismatch')
    c=json.loads(path.read_text())
    if c['schema']!='research-lossless-retirement-v1' or c['roots']!=ROOTS or c['source_script_sha256']!=sha(__file__) or c['protected_runtime_sha256']!=sha(repo/'orchestrator/codex_runtime.json'):raise ValueError('plan/source/protected runtime changed')
    if ledger.exists():raise ValueError('refusing existing execution ledger')
    begin=time.perf_counter();rows=[];state=dict(schema='research-compressed-artifacts-v1',plan_sha256=approved,entries=rows,status='IN_PROGRESS',free_bytes_before=shutil.disk_usage(repo).free)
    save(ledger,state)
    for e in c['entries']:
        if shutil.disk_usage(repo).free>=c['target_free_bytes'] or len(rows)>=c['max_files'] or time.perf_counter()-begin>c['max_wall_seconds']:break
        src=repo/e['path'];dest=repo/e['archive'];tmp=dest.with_name(dest.name+'.pending')
        if relative(repo,src)!=e['path'] or not any(src.is_relative_to(repo/r) for r in ROOTS) or src.name not in NAMES or src.is_symlink() or src.stat().st_nlink!=1:raise ValueError('source escaped allowed plan')
        if src.stat().st_size!=e['bytes'] or sha(src)!=e['sha256'] or dest.exists() or tmp.exists():raise ValueError('source changed/destination exists')
        with src.open('rb') as a,tmp.open('xb') as b:
            with gzip.GzipFile(filename='',mode='wb',fileobj=b,compresslevel=1,mtime=0) as g:shutil.copyfileobj(a,g,1024*1024)
        digest=hashlib.sha256();count=0
        with gzip.open(tmp,'rb') as f:
            while chunk:=f.read(1024*1024):digest.update(chunk);count+=len(chunk)
        if count!=e['bytes'] or digest.hexdigest()!=e['sha256'] or sha(src)!=e['sha256']:raise ValueError('compressed copy verification failed')
        tmp.replace(dest)
        row={**e,'archive_bytes':dest.stat().st_size,'archive_sha256':sha(dest),'copy_verified':True,'uncompressed_removed':False}
        rows.append(row);save(ledger,state) # Durable verified copy record BEFORE removing redundant uncompressed copy.
        src.unlink();row['uncompressed_removed']=True;save(ledger,state)
        print(json.dumps(dict(path=e['path'],saved_bytes=e['bytes']-row['archive_bytes'],free_bytes=shutil.disk_usage(repo).free)),flush=True)
    state.update(status='LOSSLESS_RETIREMENT_COMPLETE',free_bytes_after=shutil.disk_usage(repo).free,elapsed_seconds=time.perf_counter()-begin,saved_bytes=sum(e['bytes']-e['archive_bytes'] for e in rows));save(ledger,state)
    print(json.dumps({k:v for k,v in state.items() if k!='entries'}))

def restore(repo,ledger,approved,name,destination):
    if sha(ledger)!=approved:raise ValueError('ledger SHA mismatch')
    c=json.loads(ledger.read_text());matches=[e for e in c['entries'] if e['path']==name and e['copy_verified'] and e['uncompressed_removed']]
    if len(matches)!=1:raise ValueError('unknown/non-retired artifact')
    e=matches[0];src=repo/e['archive'];dest=destination or repo/name
    if sha(src)!=e['archive_sha256'] or dest.exists():raise ValueError('archive changed or destination exists')
    if shutil.disk_usage(dest.parent).free<e['bytes']+512*1024**2:raise ValueError('insufficient restore space')
    tmp=dest.with_name(dest.name+'.restore-'+uuid.uuid4().hex);digest=hashlib.sha256();count=0
    try:
        with gzip.open(src,'rb') as f,tmp.open('xb') as g:
            while chunk:=f.read(1024*1024):
                count+=len(chunk)
                if count>e['bytes']:raise ValueError('expanded length exceeded')
                digest.update(chunk);g.write(chunk)
        if count!=e['bytes'] or digest.hexdigest()!=e['sha256'] or dest.exists():raise ValueError('restore bytes changed/destination raced')
        tmp.replace(dest)
    finally:
        if tmp.exists():tmp.unlink()
    print(json.dumps(dict(status='RESTORED_VERIFIED',path=str(dest),bytes=count,sha256=e['sha256'])))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['plan','execute','restore']);a.add_argument('--plan',type=Path);a.add_argument('--sha256');a.add_argument('--ledger',type=Path);a.add_argument('--path');a.add_argument('--destination',type=Path);v=a.parse_args();repo=Path(__file__).resolve().parents[1]
    if v.action=='plan':plan(repo,v.plan)
    elif v.action=='execute':execute(repo,v.plan,v.sha256,v.ledger)
    else:restore(repo,v.ledger,v.sha256,v.path,v.destination)
if __name__=='__main__':main()
