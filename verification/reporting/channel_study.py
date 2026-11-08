"""Independent audit and bounded, readable K/accuracy plots for channel studies."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from verification.reporting.qpsk_full import audit
from verification.reconstruction.daily import sha
from verification.planning.channel_study import load_H,payload
from verification.contracts.mmse import MMSETask,task_reference
from verification.operators.catalog import adapter
from verification.profiles.numeric import policy_for


def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def plot(rows,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for channel in ('rayleigh','cdlb'):
        fig,axs=plt.subplots(2,3,figsize=(14,8))
        for i,name in enumerate(('ns','bri')):
            for j,lam in enumerate((1.,.1,.01)):
                ax=axs[i,j];selected=[r for r in rows if (r['channel'],r['algorithm'],r['regularization'])==(channel,name,lam)]
                for prec,color in [('fp16_fp32','C0'),('fp64','C1')]:
                    rr=[r for r in selected if r['precision']==prec and r['status']=='MEASURED'];bad=[r for r in selected if r['precision']==prec and r['status']!='MEASURED']
                    ax.semilogy([r['K'] for r in rr],[min(1e2,max(1e-15,r['max_error'])) for r in rr],'-o',color=color,ms=3,label=prec+' max')
                    ax.semilogy([r['K'] for r in rr],[min(1e2,max(1e-15,r['median_error'])) for r in rr],'--',color=color,alpha=.6,label=prec+' median')
                    clipped=[r for r in rr if r['max_error']>1e2]
                    if bad:ax.scatter([r['K'] for r in bad],[1e2]*len(bad),marker='x',color=color,label=prec+' rejected')
                    if clipped:ax.scatter([r['K'] for r in clipped],[1e2]*len(clipped),marker='^',color=color,label=prec+' >100')
                ax.axhline(.02,color='red',ls=':',label='2% threshold');ax.set_ylim(1e-15,3e2);ax.grid(True,which='both',alpha=.2)
                ax.set_title(f'{name.upper()}, Es/N0={-10*math.log10(lam):g} dB');ax.set_xlabel('Iterations K');ax.set_ylabel('Relative output error vs FP64 MMSE')
                ax.legend(fontsize=6,loc='lower left')
        fig.suptitle(f'{channel}: 96 fixed channels; algorithm simulation only; values >100 clipped, rejected marked x')
        fig.tight_layout();fig.savefig(out/f'{channel}_K_error.png',dpi=160);fig.savefig(out/f'{channel}_K_error.pdf');plt.close(fig)


def run(root,c,out):
    if out.exists():raise ValueError('refusing existing report')
    out.mkdir(parents=True,exist_ok=False)
    r=json.loads((root/'replay/result.json').read_text());cv=json.loads((root/'curves/result.json').read_text())
    complete=[g for g in r['groups'] if not g['rejected_frames']]
    keys={(g['operator'],g['regularization']) for g in complete}
    frames=[f for f in r['frames'] if (f['operator'],f['regularization']) in keys]
    if complete:
        checks=audit(dict(groups=complete,frames=frames,bits=sum(g['bits'] for g in complete)),root/'replay')
        checks['scope']='all complete groups, including numeric quality failures; rejected groups excluded explicitly'
    else:checks=dict(status='NO_COMPLETE_GROUPS',bits_recounted=0)
    (out/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    quality=[];diagnostics=[]
    for g in r['groups']:
        row={k:g.get(k) for k in ('operator','iterations','regularization','status','max_task_error','candidate_ber','mmse_ber','candidate_ser','mmse_ser','source_finish_cycle')}
        row['rejected_frames']=len(g['rejected_frames']);quality.append(row)
        if not g['rejected_frames']:continue
        hs=g['rejected_frames'][0]['channel_seed'];fs=g['rejected_frames'][0]['frame_seed'];t=MMSETask(**{k:g['template_task'][k] for k in MMSETask.__dataclass_fields__})
        H=load_H(root/'dataset','cdlb',hs);d=payload(t,H,hs,fs);_,_,_,fn=adapter(g['operator'],t,g['iterations']);pol=policy_for(t.profile)
        diag=dict(operator=g['operator'],regularization=t.regularization,channel_seed=hs,frame_seed=fs,
                  actual_rejection=g['rejected_frames'][0]['reason'],meaning='independent same-algorithm diagnostic only, no gate relaxation')
        # Locate an independent-reference failure without rebuilding any operator or changing its arithmetic.
        for i in range(t.batch_size):
            try:
                model=fn(H[i:i+1],d['Y'][i:i+1],t.regularization,g['iterations']);v=model['output_block'][0].astype(float);U,R=t.nt,t.nrhs
                scale=max(1.,float(np.max(abs(v))));deviation=max(float(np.max(abs(v[U:,R:]-v[:U,:R]))),float(np.max(abs(v[:U,R:]+v[U:,:R]))))/scale
                if deviation>pol.quadrant_tolerance:
                    diag.update(batch=i,independent_issue='REAL_BLOCK_QUADRANT_MISMATCH',quadrant_relative_deviation=deviation,quadrant_tolerance=pol.quadrant_tolerance);break
            except (ValueError,FloatingPointError,OverflowError) as exc:
                diag.update(batch=i,independent_issue='REFERENCE_NONFINITE_OR_ARITHMETIC',reason=str(exc));break
        else:diag['independent_issue']='NOT_REPRODUCED_IN_FIRST_FRAME_REFERENCE'
        diagnostics.append(diag)
    table(out/'quality.csv',quality);(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (out/'curve_summary.json').write_text(json.dumps([dict(channel=ch,algorithm=al,regularization=lam,precision=prec,
        first_passing_frozen_grid_K=min((q['K'] for q in cv['rows'] if (q['channel'],q['algorithm'],q['regularization'],q['precision'])==(ch,al,lam,prec) and q['passed_2pct']),default=None),
        meaning='descriptive 96-H curve; not independent holdout or minimum over untested integers') for ch in ('rayleigh','cdlb') for al in ('ns','bri') for lam in (1.,.1,.01) for prec in ('fp16_fp32','fp64')],indent=2)+'\n')
    plot(cv['rows'],out)
    (out/'manifest.json').write_text(json.dumps(dict(schema='channel-study-report-v1',replay_sha256=sha(root/'replay/result.json'),curves_sha256=sha(root/'curves/result.json'),reporter_sha256=sha(__file__),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]),indent=2)+'\n')
    print(json.dumps(checks))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--contract',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--replay-sha256',required=True);ap.add_argument('--curves-sha256',required=True);ap.add_argument('--contract-sha256',required=True);a=ap.parse_args()
    checks=[(a.root/'replay/result.json',a.replay_sha256),(a.root/'curves/result.json',a.curves_sha256),(a.contract,a.contract_sha256)]
    if any(sha(p)!=h for p,h in checks):raise ValueError('approved report input changed')
    run(a.root,json.loads(a.contract.read_text()),a.output)
    if any(sha(p)!=h for p,h in checks):raise ValueError('report input changed during audit')

if __name__=='__main__':main()
