"""Run bounded real-Asim gates. Existing inverse operator code is never changed."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from verification.asim.runner import evaluate, simulate, recover, file_hash
from verification.contracts.core import ScreenPolicy, Unsupported
from verification.fixtures.rayleigh import channel
from verification.shadow.program import Builder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--cwd',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());config['num_cores']=1
    config['core_config']={'core_0':config['core_config']['core_0']};config['precision']=8
    f=channel(42);h=f['H'];g=h.conj().T@h
    real=np.block([[h.real,-h.imag],[h.imag,h.real]])
    inputs={'a':real.T.copy(),'b':real.copy()}
    expected=np.block([[g.real,-g.imag],[g.imag,g.real]])
    b=Builder()
    for name,space,shape in [('a','DRAM',real.T.shape),('b','DRAM',real.shape),('x','SPAD',real.T.shape),('y','SPAD',real.shape),('z','ACCUM',expected.shape),('out','DRAM',expected.shape)]:b.region(name,space,shape)
    p=b.emit('LOAD',['a'],'x').emit('LOAD',['b'],'y').emit('GEMM',['x','y'],'z').emit('STORE',['z'],'out').freeze('out')
    baseline=simulate(args.binary,config,{'models':[{'name':'calibration','program':p}]},'semantic_program_test',args.cwd,root/'calibration')
    if baseline['status']!='COMPLETED':raise RuntimeError('baseline Asim failed; inspect calibration artifacts')
    cycles=baseline['finish_cycle'];policy=ScreenPolicy(cycles,1,4)
    rows=[]
    for name,program,screen,wanted in [
        ('rayleigh_gram',p,policy,'STRUCTURE_CHECKED'),
        ('not_competitive',p,ScreenPolicy(cycles*0.1,1,4),'SKIPPED_NOT_COMPETITIVE'),
        ('implausible',p,ScreenPolicy(cycles*100,1,4),'QUARANTINED_IMPLAUSIBLE_SPEEDUP')]:
        r=evaluate(program,inputs,args.binary,config,args.cwd,root/name,screen,expected)
        rows.append({'case':name,'expected_status':wanted,'status':r['status'],'numeric_status':r['numeric_status'],'cycle':r.get('asim',{}).get('finish_cycle'),'gate_passed':r['status']==wanted})
    bad=json.loads(json.dumps(p));bad['ops'].insert(3,{'id':'mutant_double','opcode':'ADD','src':['z','z'],'dst':'z'})
    # Mutants are correctness probes with a declared broad timing window, not optimization claims.
    r=evaluate(bad,inputs,args.binary,config,args.cwd,root/'wrong_arithmetic',ScreenPolicy(cycles,0.01,100),expected)
    rows.append({'case':'wrong_arithmetic','expected_status':'NUMERIC_FAILED','status':r['status'],'numeric_status':r['numeric_status'],'relative_error':r.get('relative_error'),'gate_passed':r['status']=='NUMERIC_FAILED'})
    missing=json.loads(json.dumps(p));missing['ops'].pop(1)
    r=evaluate(missing,inputs,args.binary,config,args.cwd,root/'missing_load',policy,expected)
    rows.append({'case':'missing_load','expected_status':'SKIPPED_ASIM_FAILED','status':r['status'],'numeric_status':r['numeric_status'],'gate_passed':r['status']=='SKIPPED_ASIM_FAILED' and r['numeric_status']=='NOT_RUN'})
    for mode,name,m,u in [('newton_schulz_v3_test','legacy_ns',4,4),('cholesky_noblock_v2_test','legacy_cholesky',4,2)]:
        legacy=json.loads(json.dumps(config));legacy['precision']=2
        receipt=simulate(args.binary,legacy,{'models':[{'name':name,'batch_size':1,'matrix_m':m,'matrix_k':u}]},mode,args.cwd,root/name)
        result={'case':name,'asim_status':receipt['status'],'cycle':receipt['finish_cycle'],'score':None,'numeric_status':'NOT_RUN'}
        if receipt['status']=='COMPLETED':
            records=[json.loads(s) for s in (root/name/'instructions.jsonl').read_text().splitlines()]
            result['actual_instruction_count']=sum(len(x['instructions']) for x in records)
            try:recover(records[0],{})
            except Unsupported as error:result.update(status='UNSUPPORTED',reason=str(error),gate_passed=True)
            else:result.update(status='UNEXPECTED_LEGACY_ACCEPTANCE',gate_passed=False)
        else:result.update(status='ASIM_FAILED',gate_passed=False)
        (root/name/'reconstruction.json').write_text(json.dumps(result,indent=2));rows.append(result)
    sources=[]
    source_root=Path(__file__).resolve().parents[2]
    for base in ('src/semantic','verification'):
        for path in sorted((source_root/base).rglob('*')):
            if path.suffix in ('.py','.h','.cc'):sources.append({'path':str(path.relative_to(source_root)),'sha256':file_hash(path)})
    summary={'status':'GATES_PASSED' if all(r['gate_passed'] for r in rows) else 'GATES_FAILED','scope':'structure prototype; no G4/precision/quality/promotion certification','score':None,'cases':rows,'binary_sha256':file_hash(args.binary),'binary_source_commit':subprocess.check_output(['git','-C',str(args.cwd),'rev-parse','HEAD'],text=True).strip(),'runner_base_commit':subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip(),'sources':sources,'artifacts':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ('artifacts','sources')},ensure_ascii=False))
    return 0 if summary['status']=='GATES_PASSED' else 1

if __name__=='__main__':raise SystemExit(main())
