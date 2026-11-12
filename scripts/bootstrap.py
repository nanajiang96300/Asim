"""Pinned Linux x86_64 dependency installation and independent source build (stdlib only)."""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def lock():
    p=ROOT/'dependencies/lock.json';d=json.loads(p.read_text())
    if d.get('schema')!='asim-dependencies-v1':raise ValueError('unsupported dependency lock')
    return d,digest(p)
def safe_path(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or not name:raise ValueError('unsafe dependency path')
    return p
def verify(out,spec,identity):
    receipt=json.loads((out/'INSTALL_RECEIPT.json').read_text())
    if receipt.get('lock_sha256')!=identity:raise ValueError('installed dependency lock mismatch; use a separate dependency directory')
    manifest=json.loads((out/'DEPENDENCY_MANIFEST.json').read_text())
    if digest(out/'DEPENDENCY_MANIFEST.json')!=spec['source_manifest_sha256']:raise ValueError('dependency manifest changed')
    names=set()
    entries=manifest['files']+[{'path':'vendor/wheels/'+e['file'],'bytes':e['bytes'],'sha256':e['sha256']} for e in spec['artifacts'] if e['kind']=='wheel']
    for e in entries:
        p=safe_path(e['path']);f=out/p
        if str(p) in names:raise ValueError('duplicate dependency file')
        names.add(str(p))
        if f.is_symlink() or not f.is_file() or f.stat().st_size!=e['bytes'] or digest(f)!=e['sha256']:raise ValueError('dependency missing/tampered: '+str(p))
    expected=names|{'DEPENDENCY_MANIFEST.json','INSTALL_RECEIPT.json'}
    for p in out.rglob('*'):
        if p.is_symlink() or (p.is_file() and str(p.relative_to(out)) not in expected):raise ValueError('unexpected dependency file: '+str(p.relative_to(out)))
    return len(entries)
def artifact(e,cache,offline):
    if Path(e['file']).name!=e['file'] or e['file'] in ('','.','..'):raise ValueError('unsafe artifact filename')
    cache.mkdir(parents=True,exist_ok=True);p=cache/e['file']
    if p.exists():
        if p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:raise ValueError('cached dependency SHA mismatch: '+p.name)
        return p
    if offline:raise ValueError('offline cache missing: '+p.name)
    if not e['url'].startswith('https://raw.githubusercontent.com/nanajiang96300/Asim/'):raise ValueError('unapproved dependency origin')
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=cache,prefix='.download-',delete=False) as f:
            tmp=Path(f.name);size=0
            with urllib.request.urlopen(e['url'],timeout=30) as response:
                while chunk:=response.read(1024*1024):
                    size+=len(chunk)
                    if size>e['bytes']:raise ValueError('dependency exceeds declared size')
                    f.write(chunk)
        if tmp.stat().st_size!=e['bytes'] or digest(tmp)!=e['sha256']:raise ValueError('downloaded dependency SHA mismatch')
        tmp.replace(p);return p
    finally:
        if tmp and tmp.exists():tmp.unlink()
def prepare(out,spec,identity,cache,offline):
    if out.exists():return verify(out,spec,identity)
    assets={e['file']:artifact(e,cache,offline) for e in spec['artifacts']};source=next(e for e in spec['artifacts'] if e['kind']=='cpp-source')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.deps-stage-') as name:
        stage=Path(name)/'payload';stage.mkdir()
        with tarfile.open(assets[source['file']],'r:gz') as t:
            members=t.getmembers();seen=set()
            if len(members)>20000 or sum(m.size for m in members)>300*1024**2:raise ValueError('dependency archive exceeds limits')
            for m in members:
                p=safe_path(m.name)
                if not m.isfile() or str(p) in seen or p.parts[0] not in ('vendor','generated','DEPENDENCY_MANIFEST.json'):raise ValueError('unsupported archive member')
                seen.add(str(p))
            t.extractall(stage,filter='data')
        wheels=stage/'vendor/wheels';wheels.mkdir()
        for e in spec['artifacts']:
            if e['kind']=='wheel':shutil.copyfile(assets[e['file']],wheels/e['file'])
        (stage/'INSTALL_RECEIPT.json').write_text(json.dumps({'schema':'asim-installed-dependencies-v1','lock_sha256':identity,'dependency_commit':spec['dependency_commit']},indent=2)+'\n')
        n=verify(stage,spec,identity)
        if out.exists():raise ValueError('dependency directory created concurrently; refusing overwrite')
        stage.rename(out)
    return n
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--cache',type=Path,default=ROOT/'.deps/cache');ap.add_argument('--deps-dir',type=Path,default=ROOT/'.deps/asim-deps-v1');ap.add_argument('--offline',action='store_true');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--verify-only',action='store_true');ap.add_argument('--download-only',action='store_true');ap.add_argument('--build-dir',type=Path,default=ROOT/'build');ap.add_argument('--jobs',type=int,default=1);a=ap.parse_args()
    spec,identity=lock();out=a.deps_dir.resolve()
    if a.verify_only:n=verify(out,spec,identity)
    elif a.download_only:
        for e in spec['artifacts']:artifact(e,a.cache.resolve(),a.offline)
        print('DEPENDENCY_CACHE_CHECKED');return
    else:n=prepare(out,spec,identity,a.cache.resolve(),a.offline)
    print('DEPENDENCIES_CHECKED',n,flush=True)
    if a.prepare_only or a.verify_only:return
    if sys.version_info[:2]!=(3,12) or sys.platform!='linux' or platform.machine()!='x86_64':raise ValueError('bundled wheels require Linux x86_64 / CPython3.12')
    if a.jobs<1:raise ValueError('jobs must be positive')
    venv=ROOT/'.venv';subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True);py=venv/'bin/python';wheels=out/'vendor/wheels'
    env=os.environ.copy();env['PYTHONPATH']=str(wheels/'pip-26.1.2-py3-none-any.whl')
    subprocess.run([str(py),'-m','pip','install','--no-index','--find-links',str(wheels),'pip==26.1.2','numpy==2.5.0'],env=env,check=True)
    subprocess.run(['cmake','-S',str(ROOT),'-B',str(a.build_dir.resolve()),'-DCMAKE_BUILD_TYPE=Release','-DASIM_DEPS_DIR='+str(out)],check=True)
    subprocess.run(['cmake','--build',str(a.build_dir.resolve()),'--target','Simulator','-j'+str(a.jobs)],check=True)
if __name__=='__main__':main()
