"""Export a clean Claude Git version with offline assets, no history/results/auth/builds."""
import argparse,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOP={'CMakeLists.txt','LICENSE','LICENSE.md','README.md','BUILDING.md','AGENTS.md','CLAUDE.md','.gitignore'}
PREFIX={'src','configs','verification','scripts','tests','orchestrator','agent_profiles','dependencies','packaging'}
EXT={'booksim','ramulator_custom','ramulator2','onnx','protobuf'}
DATA={'.npz','.npy','.onnx','.pt','.pth','.pkl','.pickle','.trace','.log','.csv','.pdf','.png','.jpg','.gif','.zip','.gz','.whl','.bin','.so','.a','.o','.pyc'}

def selected(name):
    p=Path(name)
    if p.suffix.lower() in DATA or any(n in {'__pycache__','testdata','data','benchmarks','perf_comparison','rh_study'} for n in p.parts):return False
    if len(p.parts)==1:return name in TOP
    if p.parts[0] in PREFIX:return True
    if p.parts[0]=='extern':return len(p.parts)>2 and p.parts[1] in EXT
    if p.parts[:2]==('.claude','skills'):return True
    if p.parts[0]=='DOCS':
        return p.suffix=='.md' and (p.name in {'CLAUDE_CODE_USER_GUIDE.md','OPERATOR_DEVELOPMENT_STANDARD_V3.md','DAG_PRIMITIVES_SPEC.md','NEW_OPERATOR_CHECKLIST.md'} or len(p.parts)>2 and p.parts[1] in {'research','operators','specs'} and not any(n in {'paper_archive','paper_reports'} for n in p.parts))
    return False

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--cache',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--name',default='asim-claude-guide-v1');v=a.parse_args()
    if Path(v.name).name!=v.name or v.name in ('.','..'):raise ValueError('invalid package name')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('commit tracked changes first')
    profile=json.loads((ROOT/'agent_profiles/active.json').read_text())
    if profile.get('client')!='claude':raise ValueError('export from the Claude client branch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();parent=v.output_root.resolve();out=parent/v.name;archive=parent/(v.name+'.tar.gz')
    if out.exists() or archive.exists():raise ValueError('refusing existing source package or archive')
    lock=json.loads((ROOT/'dependencies/lock.json').read_text())
    # Check all supplied assets before creating the output.
    for e in lock['artifacts']:
        p=v.cache/e['file']
        if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise ValueError('bad offline asset: '+e['file'])
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0');out.mkdir(parents=True)
    included=[];excluded=[]
    for name in names:
        if not name:continue
        if not selected(name):excluded.append(name);continue
        src=ROOT/name
        if src.is_symlink():raise ValueError('source symlink needs explicit audit: '+name)
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);included.append(name)
    # Replace only one historical fixture dependency with a synthetic test contract.
    source=json.loads((ROOT/'experiments/qpsk_full/2026-11-08-v1/source-contract.json').read_text());source['runtime']={'path':'not-used-in-this-unit-test','sha256':'example-only'};source['producer_sources']={k:'example-only' for k in source['producer_sources']}
    for op in source['algorithms']:op['source']={'path':'not-used-in-this-unit-test','sha256':'example-only'}
    p=out/'verification/tests/fixtures/qpsk-source-contract.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(source,indent=2)+'\n')
    p=out/'verification/tests/test_qpsk_link.py';s=p.read_text();old="Path('experiments/qpsk_full/2026-11-08-v1/source-contract.json')";assert s.count(old)==1;s=s.replace(old,"(Path(__file__).parent/'fixtures/qpsk-source-contract.json')");p.write_text(s)
    (out/'offline_cache').mkdir()
    for e in lock['artifacts']:shutil.copy2(v.cache/e['file'],out/'offline_cache'/e['file'])
    shutil.copy2(out/'DOCS/CLAUDE_CODE_USER_GUIDE.md',out/'使用说明.md')
    provenance={'schema':'asim-claude-source-delivery-v1','source_commit':commit,'source_branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'target':'Linux x86_64 CPython3.12; WSL2 not tested','profile':profile,'overlays':['one unit-test synthetic fixture and path; no replay/runtime changes','root Chinese guide is a byte-identical alias of DOCS/CLAUDE_CODE_USER_GUIDE.md'],'cpp_sources_unchanged':True,'excluded_categories':['Git history','experiment results and datasets','build and venv','credentials/settings/sessions','unrelated external tools and paper snapshots'],'excluded_source_paths':excluded}
    (out/'SOURCE_VERSION.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'BUNDLE_MANIFEST.json').write_text(json.dumps({'schema':'asim-delivery-files-v1','source_commit':commit,'files':rows},indent=2)+'\n')
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(out,arcname=v.name,recursive=True)
    (parent/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'status':'PACKED_NOT_YET_VALIDATED','source_commit':commit,'directory':str(out),'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(rows)}))
if __name__=='__main__':main()
