"""Plots and compact tables for the frozen FP16 primitive feasibility study."""
import argparse,csv,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verification.reconstruction.daily import sha


def report(root,out):
    if out.exists():raise ValueError('refusing existing report')
    cp=root/'2026-11-08-v3/calibration/result.json';hp=root/'2026-11-08-v3/holdout/result.json';ap=root/'2026-11-08-v3/independent_audit_v2/result.json'
    c=json.loads(cp.read_text());h=json.loads(hp.read_text());a=json.loads(ap.read_text())
    if a['result_sha256']!=sha(hp) or a['status']!='FP16_PRIMITIVE_HOLDOUT_CHECKED':raise ValueError('unverified holdout')
    out.mkdir(parents=True,exist_ok=False);fig,axs=plt.subplots(2,2,figsize=(13,9))
    colors=['C0','C1','C2']
    for lam,color in zip([1.,.1,.01],colors):
        v=next(v for v in c['cases'] if v['kind']=='implicit_solve' and v['block']==1 and v['lambda']==lam)
        axs[0,0].semilogy([r['K'] for r in v['rows']],[r['task_error_max'] for r in v['rows']],color=color,label=f'lambda={lam}')
        held=next(x for x in a['rows'] if x['goal']=='mmse_output' and x['regularization']==lam)
        rr=next(x for x in v['rows'] if x['K']==held['K']);axs[0,0].scatter([rr['K']],[rr['task_error_max']],marker='*',s=90,color=color)
    axs[0,0].set_title('Calibration96: direct MMSE solve, complex block1')
    for b,color in zip([1,4,8],colors):
        v=next(v for v in c['cases'] if v['kind']=='implicit_inverse_add' and v['block']==b)
        axs[0,1].semilogy([r['K'] for r in v['rows']],[r['inverse_error_max'] for r in v['rows']],color=color,label=f'complex block={b}')
    axs[0,1].set_title('Calibration96: inverse error, Es/N0=20dB')
    for kind,label,color in [('implicit_solve','direct solve','C0'),('implicit_inverse_add','inverse then apply','C3')]:
        v=next(v for v in c['cases'] if v['kind']==kind and v['block']==1 and v['lambda']==.01)
        axs[1,0].semilogy([r['K'] for r in v['rows']],[r['task_error_max'] for r in v['rows']],color=color,label=label)
    axs[1,0].set_title('Calibration96: two MMSE paths, Es/N0=20dB')
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:
        ax.axhline(.02,color='black',ls=':',label='2% acceptance');ax.axhline(.01,color='gray',ls='--',label='1% calibration margin')
        ax.set_xlabel('K');ax.set_ylabel('Maximum relative error');ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7)
    vals=[100*r['max_goal_error'] for r in a['rows']]+[100*a['rows'][-1]['inverse_then_apply_task_error_max']]
    labels=['MMSE0','MMSE10','MMSE20','Inverse20','Inv-apply20']
    ax=axs[1,1];ax.bar(labels,vals,color=['C2']*4+['C3']);ax.axhline(2,color='black',ls=':')
    for i,v in enumerate(vals):ax.text(i,v+.15,f'{v:.3f}%',ha='center',fontsize=8)
    ax.set_ylim(0,max(vals)*1.15);ax.set_ylabel('Maximum relative error (%)');ax.set_title('Independent holdout192: distinct frozen goals')
    fig.suptitle('FP16 tensors + FP16 compensation; declared FP32 temporary arithmetic; no Asim run')
    fig.tight_layout();fig.savefig(out/'FP16_feasibility.png',dpi=160);fig.savefig(out/'FP16_feasibility.pdf');plt.close(fig)
    with (out/'calibration_curves.csv').open('w',newline='') as f:
        rows=[dict(case=v['case'],regularization=v['lambda'],block=v['block'],kind=v['kind'],**r) for v in c['cases'] for r in v['rows']]
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with (out/'holdout.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(a['rows'][0]),lineterminator='\n');w.writeheader();w.writerows(a['rows'])
    m=dict(schema='fp16-feasibility-report-v1',reporter_sha256=sha(__file__),inputs=[dict(path=str(p),sha256=sha(p)) for p in [cp,hp,ap]],outputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())])
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();report(a.root,a.output)
