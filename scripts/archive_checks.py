"""Strict immutable artifact-manifest helpers for audit tools."""
import hashlib,re
from pathlib import Path

CASE_REQUIRED={'asim.json','config.json','models.json','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','schedule.json','inputs.npz','references.npz','outputs.npy','simulator.log'}
PROCESS_REQUIRED={'config.json','models.json','instructions.jsonl','dispatch.jsonl','trace.csv','rw_events.jsonl','simulator.log'}

def indexed(entries,required=()):
    result={}
    for e in entries:
        path=e['path'];p=Path(path)
        if not isinstance(path,str) or p.is_absolute() or '..' in p.parts or str(p)!=path or path in ('','.'):raise ValueError('unsafe artifact path')
        if path in result:raise ValueError('duplicate artifact path')
        if type(e['bytes']) is not int or e['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',e['sha256']):raise ValueError('invalid artifact digest')
        result[path]=e
    if not set(required)<=result.keys():raise ValueError('missing required artifacts')
    return result

def nested(outer,prefix,entries,required=()):
    inner=indexed(entries,required)
    for path,e in inner.items():
        key=str(Path(prefix)/path)
        if outer.get(key)!={**e,'path':key}:raise ValueError('nested artifact digest differs from outer manifest')
    return inner

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify_files(root,index):
    for rel,e in index.items():
        p=Path(root)/rel
        if not p.is_file() or p.stat().st_size!=e['bytes'] or sha_file(p)!=e['sha256']:raise ValueError('artifact changed: '+str(p))
