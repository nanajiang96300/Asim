"""Source-only expert distribution. Explicit inputs; no research data/results/auth."""
from pathlib import Path
import shutil,subprocess,json,hashlib,sys,argparse
cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("--wheel-dir",type=Path,required=True,help="directory with pinned pip26.1.2 and NumPy2.5.0 CPython3.12 Linux wheels");args=cli.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'expert_delivery/asim-expert-v1';assert not OUT.exists();OUT.mkdir(parents=True)
SKIP={'.git','__pycache__','.pytest_cache','node_modules','testdata','data','bench','benchmarks','examples','docs','doc','logos','test','tests'}
def copy(src,dst,extensions=None):
 if src.is_dir():
  dst.mkdir(parents=True,exist_ok=True)
  for p in sorted(src.iterdir()):
   if p.name in SKIP or p.is_symlink():continue
   copy(p,dst/p.name,extensions)
 elif src.is_file() and (extensions is None or src.suffix in extensions or src.name in ['CMakeLists.txt','LICENSE','NOTICE','COPYING','VERSION_NUMBER','LICENSE_1_0.txt','INSTALL']):
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
for name in ['src','configs']:
 copy(ROOT/name,OUT/name)
# Copy full verification code including tests, but no bytecode, arrays or logs.
shutil.copytree(ROOT/'verification',OUT/'verification',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(OUT/'orchestrator').mkdir();shutil.copy2(ROOT/'orchestrator/codex_operators.json',OUT/'orchestrator/codex_operators.json')
for name in ['booksim','ramulator_custom','ramulator2']:
 copy(ROOT/'extern'/name,OUT/'extern'/name)
# ramulator/src/test is compiled simulator code, not an experiment dataset.
shutil.copytree(ROOT/'extern/ramulator2/src/test',OUT/'extern/ramulator2/src/test')
for name in ['perf_comparison','rh_study']:
 p=OUT/'extern/ramulator2'/name
 if p.exists():shutil.rmtree(p)
for p in (OUT/'extern').rglob('*.trace'):p.unlink()
# Only headers and protobuf schema sources from ONNX, not its test models/data.
copy(ROOT/'extern/onnx/onnx',OUT/'extern/onnx/onnx',{'.h','.hpp','.inc','.proto','.proto3'})
for name in ['LICENSE','VERSION_NUMBER']:copy(ROOT/'extern/onnx'/name,OUT/'extern/onnx'/name)
copy(ROOT/'extern/protobuf/src',OUT/'extern/protobuf/src',{'.h','.hpp','.inc','.cc','.c','.proto','.in','.map','.am'})
copy(ROOT/'extern/protobuf/cmake',OUT/'extern/protobuf/cmake')
for name in ['configure.ac','LICENSE','README.md']:copy(ROOT/'extern/protobuf'/name,OUT/'extern/protobuf'/name)
gen=Path('/home/nanajiang/asim-worktrees/bri-fixed-hb-20261109/build/extern/onnx/onnx')
for p in gen.glob('*.pb.*'):
 if p.suffix in ['.cc','.h']:copy(p,OUT/'generated/onnx'/p.name)
boost=Path('/home/nanajiang/.conan/data/boost/1.79.0/_/_/source/src')
copy(boost/'boost',OUT/'vendor/include/boost')
copy(boost/'libs/program_options/src',OUT/'vendor/boost_program_options/src')
copy(boost/'LICENSE_1_0.txt',OUT/'vendor/boost_program_options/LICENSE_1_0.txt')
copy(boost/'LICENSE_1_0.txt',OUT/'vendor/include/boost/LICENSE_1_0.txt')
conan=Path('/home/nanajiang/.conan/data')
for lib,ver in [('robin-hood-hashing','3.11.5'),('nlohmann_json','3.11.2')]:
 pkg=next((conan/lib/ver/'_/_/package').iterdir());copy(pkg/'include',OUT/'vendor/include')
 copy(pkg/'licenses',OUT/'vendor/licenses'/lib)
cache=Path('/home/nanajiang/asim-worktrees/fw-asim-gate-20261029/build')
for name,src in [('spdlog',cache/'ext/spdlog'),('argparse',cache/'ext/argparse')]:
 copy(src/'include',OUT/'vendor'/name/'include');copy(src/'LICENSE',OUT/'vendor'/name/'LICENSE')
copy(cache/'ext/spdlog/src/bundled_fmtlib_format.cpp',OUT/'vendor/spdlog/src/bundled_fmtlib_format.cpp')
yaml=cache/'_deps/yaml-cpp-src';copy(yaml,OUT/'vendor/yaml-cpp')
# Source-vendored Ramulator dependency setup; retain original model sources.
r=OUT/'extern/ramulator2/CMakeLists.txt';s=r.read_text();tail=s[s.index('include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)'):];r.write_text('cmake_minimum_required(VERSION 3.20)\nproject(Ramulator LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 20)\nset(CMAKE_POSITION_INDEPENDENT_CODE ON)\n'+tail)
shutil.copy2(ROOT/'packaging/templates/CMakeLists.txt',OUT/'CMakeLists.txt')
# Adapt one historical fixture to a standalone contract-only unit-test fixture.
test=OUT/'verification/tests/test_qpsk_link.py';text=test.read_text().replace("Path('experiments/qpsk_full/2026-11-08-v1/source-contract.json')","Path(__file__).parent/'fixtures/source-contract.json'")
# parentheses are needed before .read_text on the Path expression.
text=text.replace("json.loads(Path(__file__).parent/'fixtures/source-contract.json'.read_text())","json.loads((Path(__file__).parent/'fixtures/source-contract.json').read_text())")
test.write_text(text)
fixture=json.loads((ROOT/'experiments/qpsk_full/2026-11-08-v1/source-contract.json').read_text());fixture['runtime']={'path':'not-used-in-this-unit-test','sha256':'example-only'};fixture['producer_sources']={k:'example-only' for k in fixture['producer_sources']}
for a in fixture['algorithms']:a['source']={'path':'not-used-in-this-unit-test','sha256':'example-only'}
p=OUT/'verification/tests/fixtures/source-contract.json';p.parent.mkdir();p.write_text(json.dumps(fixture,indent=2)+'\n')
# Original current workflow skills, no personal settings, accounts or sessions.
for name in ['op-flow','op-dev','audit-operator']:
 copy(ROOT/'.claude/skills'/name,OUT/'.claude/skills'/name)
copy(ROOT/'DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md',OUT/'DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md')
(OUT/'scripts').mkdir();(OUT/'docs').mkdir();(OUT/'example').mkdir()
from verification.contracts.mmse import MMSETask
from verification.operators.catalog import adapter
ops=[('cholesky_explicit_v1',1),('ldl_explicit_v1',1),('cholesky_block_explicit_v1',1),('ldl_block_explicit_v1',1),('ns_corrected_v1',16),('bri_explicit_v1',192)]
for op,k in ops:
 _,_,models,_=adapter(op,MMSETask(profile='modeled_fp16_fp32_real_block_v1'),k);(OUT/'example'/f'{op}.json').write_text(json.dumps(models,indent=2)+'\n')
meta=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),target='Linux x86_64 / WSL2 Ubuntu',packaging_changes=['portable source-vendored CMake','Ramulator dependency setup only','one unit-test contract fixture path; no historical results'],cpp_operator_sources_unchanged=True,dependencies={'boost':'1.79.0','robin_hood':'3.11.5','nlohmann_json':'3.11.2','spdlog':'1.11.0 headers + bundled fmt implementation source','yaml-cpp':'0.7.0','argparse':'2.9','onnx':(ROOT/'extern/onnx/VERSION_NUMBER').read_text().strip(),'protobuf':'vendored runtime with matching generated ONNX protobuf C++ sources'},not_included=['datasets','experiment results','model weights','server binaries/build caches','Git history','credentials or account/session settings'])
meta['packaging_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
meta['source_cpp_inventory']=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/'src').rglob('*')) if p.is_file()]
(OUT/'SOURCE_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(package=str(OUT),files=sum(p.is_file() for p in OUT.rglob('*')))))

for pattern in ["pip-26.1.2-py3-none-any.whl","numpy-2.5.0-cp312-cp312-manylinux*x86_64.whl"]:
 matches=list(args.wheel_dir.glob(pattern));assert len(matches)==1,(pattern,matches);copy(matches[0],OUT/"vendor/wheels"/matches[0].name)
subprocess.run([sys.executable,str(ROOT/"packaging/finalize.py")],check=True,cwd=ROOT)
