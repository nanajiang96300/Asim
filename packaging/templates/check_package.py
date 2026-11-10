"""Check shipped source files; generated build/work files are outside the manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'PACKING_MANIFEST.json').read_text())
for e in m['files']:
 p=Path(e['path'])
 if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe path')
 f=root/p
 if f.is_symlink() or not f.is_file() or f.stat().st_size!=e['bytes']:raise ValueError('missing/size/symlink mismatch: '+str(p))
 with f.open('rb') as h:d=hashlib.file_digest(h,'sha256').hexdigest()
 if d!=e['sha256']:raise ValueError('SHA mismatch: '+str(p))
print('SOURCE_MANIFEST_CHECKED',len(m['files']))
