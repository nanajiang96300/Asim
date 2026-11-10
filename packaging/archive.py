"""Archive only a sealed source package and verify a fresh extraction + tamper gate."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,tempfile
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'expert_delivery';name='asim-expert-v1';source=base/name
subprocess.run([sys.executable,str(source/'scripts/check_package.py')],check=True)
archive=base/(name+'.tar.gz')
subprocess.run(['tar','--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner','-czf',str(archive),'-C',str(base),name],check=True)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers()
 for e in members:
  p=Path(e.name)
  if p.is_absolute() or '..' in p.parts or p.parts[0]!=name or not (e.isdir() or e.isfile()):raise ValueError('unsafe member: '+e.name)
  if set(p.parts)&{'build','.venv','__pycache__','results','validation-work'}:raise ValueError('generated directory: '+e.name)
  if p.suffix in {'.npy','.npz','.log','.trace','.onnx','.jsonl','.pyc','.o','.a','.so'}:raise ValueError('data/build file: '+e.name)
 with tempfile.TemporaryDirectory(prefix='extract-check-',dir=base) as scratch:
  t.extractall(scratch,filter='data');r=Path(scratch)/name
  subprocess.run([sys.executable,str(r/'scripts/check_package.py')],check=True)
  p=r/'README.md';p.write_text(p.read_text()+'\nintentional integrity negative test\n')
  n=subprocess.run([sys.executable,str(r/'scripts/check_package.py')],capture_output=True,text=True)
  if n.returncode==0:raise ValueError('manifest failed to reject tamper')
with archive.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
(archive.with_suffix(archive.suffix+'.sha256')).write_text(d+'  '+archive.name+'\n')
report=dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=d,members=len(members),fresh_extraction_manifest='PASS',tampered_readme_rejected=True,no_data_results_build_or_symlinks=True)
p=ROOT/'experiments/distribution/2026-11-10-v1/archive_check.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
