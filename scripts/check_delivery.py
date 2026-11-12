"""Check a fresh source delivery against its per-file size/SHA manifest (stdlib only)."""
import argparse,hashlib,json
from pathlib import Path

def check(root):
    root=Path(root).resolve();manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text());seen=set()
    for e in manifest['files']:
        p=Path(e['path'])
        if p.is_absolute() or '..' in p.parts or str(p) in seen:raise ValueError('unsafe/duplicate manifest path')
        seen.add(str(p));f=root/p
        if not f.resolve().is_relative_to(root) or f.is_symlink() or not f.is_file():raise ValueError('missing/unsafe file: '+str(p))
        with f.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
        if f.stat().st_size!=e['bytes'] or h!=e['sha256']:raise ValueError('changed file: '+str(p))
    runtime={'.git','.deps','.venv','build','results','__pycache__'}
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if any(n in runtime for n in rel.parts) or p.suffix=='.pyc':continue
        if p.is_symlink() or p.is_file() and str(rel) not in seen|{'BUNDLE_MANIFEST.json'}:raise ValueError('unexpected file: '+str(rel))
    return {'status':'DELIVERY_FILES_CHECKED','files':len(seen),'source_commit':manifest['source_commit']}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);v=a.parse_args();print(json.dumps(check(v.root)))
if __name__=='__main__':main()
