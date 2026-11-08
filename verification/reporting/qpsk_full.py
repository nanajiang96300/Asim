"""Independently audit archived bit decisions and export paper-ready QPSK artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(result,root):
    from scipy.stats import binomtest
    labels=np.array([[1,1],[1,0],[0,1],[0,0]],np.uint8)
    points=((2*labels[:,0].astype(float)-1)+1j*(2*labels[:,1].astype(float)-1))/math.sqrt(2)
    def decode(z):return labels[np.argmin(np.abs(z[...,None]-points)**2,axis=-1)]
    groups={(g['operator'],g['regularization']):g for g in result['groups']};acc={};cache={}
    for f in result['frames']:
        ia=f['input_artifact'];oa=f['output_artifact'];ip=root/ia['path'];op=root/oa['path']
        for p,a in [(ip,ia),(op,oa)]:
            if p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:raise ValueError('changed frame artifact: '+str(p))
        if str(ip) not in cache:
            with np.load(ip,allow_pickle=False) as n:cache[str(ip)]={k:n[k] for k in ['H','Y','bits','symbols','noise','mmse']}
        data=cache[str(ip)];bits=data['bits'];B,U,R=bits.shape[:3]
        bf=bits.astype(float);mapped=((2*bf[...,0]-1)+1j*(2*bf[...,1]-1))/math.sqrt(2)
        if not np.allclose(mapped,data['symbols'],rtol=0,atol=1e-14) or not np.allclose(data['Y'],data['H']@mapped+data['noise'],rtol=1e-12,atol=1e-12):
            raise ValueError('independent modulation/channel equation audit failed')
        with np.load(op,allow_pickle=False) as n:raw=n['actual_output_block'].astype(float)
        if raw.shape!=(B,2*U,2*R):raise ValueError('output coverage mismatch')
        x=raw[:,:U,:R]+1j*raw[:,U:,:R]
        H,Y=data['H'],data['Y'];Ht=H.conj().transpose(0,2,1)
        target=np.linalg.solve(Ht@H+f['regularization']*np.eye(U),Ht@Y)
        if not np.allclose(target,data['mmse'],rtol=1e-12,atol=1e-12):raise ValueError('archived MMSE mismatch')
        rx=decode(x);ref=decode(target);a=rx!=bits;b=ref!=bits;sa=np.any(a,axis=-1);sb=np.any(b,axis=-1)
        k=(f['operator'],f['regularization'])
        if k not in acc:acc[k]=dict(a=0,b=0,sa=0,sb=0,bits=0,symbols=0,channels={},ca=np.zeros((4,4),int),cb=np.zeros((4,4),int))
        v=acc[k];v['a']+=int(a.sum());v['b']+=int(b.sum());v['sa']+=int(sa.sum());v['sb']+=int(sb.sum());v['bits']+=bits.size;v['symbols']+=B*U*R
        txid=2*bits[...,0]+bits[...,1]
        for name,d in [('ca',rx),('cb',ref)]:v[name]+=np.bincount((4*txid+2*d[...,0]+d[...,1]).reshape(-1),minlength=16).reshape(4,4)
        for i in range(B):
            hk=(f['channel_seed'],i)
            if hk not in v['channels']:v['channels'][hk]=dict(a=0,b=0,sa=0,sb=0,bits=0,symbols=0,ea=0.,eb=0.)
            q=v['channels'][hk];q['a']+=int(a[i].sum());q['b']+=int(b[i].sum());q['sa']+=int(sa[i].sum());q['sb']+=int(sb[i].sum());q['bits']+=bits[i].size;q['symbols']+=U*R
            q['ea']+=float(np.sum(np.abs(x[i]-data['symbols'][i])**2));q['eb']+=float(np.sum(np.abs(target[i]-data['symbols'][i])**2))
    for k,v in acc.items():
        g=groups[k]
        for field,key in [('candidate_bit_errors','a'),('mmse_bit_errors','b'),('candidate_symbol_errors','sa'),('mmse_symbol_errors','sb'),('bits','bits'),('symbols','symbols')]:
            if g[field]!=v[key]:raise ValueError('independent count mismatch: '+str(k)+' '+field)
        if g['candidate_confusion']!=v['ca'].tolist() or g['mmse_confusion']!=v['cb'].tolist():raise ValueError('confusion mismatch')
        for row in g['channels']:
            q=v['channels'][(row['channel_seed'],row['batch'])]
            for name,key in [('candidate_bit_errors','a'),('mmse_bit_errors','b'),('candidate_symbol_errors','sa'),('mmse_symbol_errors','sb')]:
                if row[name]!=q[key]:raise ValueError('per-H count mismatch')
            if abs(row['candidate_raw_evm']-math.sqrt(q['ea']/q['symbols']))>1e-12 or abs(row['mmse_raw_evm']-math.sqrt(q['eb']/q['symbols']))>1e-12:
                raise ValueError('per-H EVM mismatch')
        for bound in g['outage_bounds'].values():
            ref=binomtest(bound['count'],bound['independent_channels']).proportion_ci(confidence_level=1-2*g['tail_alpha'],method='exact')
            if abs(ref.low-bound['lower'])>1e-11 or abs(ref.high-bound['upper'])>1e-11:raise ValueError('independent confidence-bound mismatch')
    if len(acc)!=len(groups) or sum(v['bits'] for v in acc.values())!=result['bits']:raise ValueError('incomplete independent audit')
    return dict(status='INDEPENDENT_QPSK_AUDIT_PASSED',groups=len(acc),frame_records=len(result['frames']),bits_recounted=sum(v['bits'] for v in acc.values()),
                method='minimum-distance demap, recomputed FP64 MMSE, exact bit/symbol/confusion/per-H EVM counts; independent SciPy CP intervals')


def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(path,expected,output):
    if sha(path)!=expected:raise ValueError('approved link result changed')
    root=path.parent;r=json.loads(path.read_text());c=json.loads((root/'contract.json').read_text())
    if not r.get('uncoded_qpsk_suite_complete'):raise ValueError('complete test result required')
    if output.exists():raise ValueError('refusing existing reporting output')
    checks=audit(r,root);output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    rows=[];streams=[]
    for g in r['groups']:
        rows.append(dict(operator=g['operator'],iterations=g['iterations'],es_no_db=g['es_no_db'],eb_no_db=g['eb_no_db'],noise_variance=g['regularization'],
                         H=g['independent_channels'],bits=g['bits'],symbols=g['symbols'],candidate_bit_errors=g['candidate_bit_errors'],mmse_bit_errors=g['mmse_bit_errors'],
                         candidate_symbol_errors=g['candidate_symbol_errors'],mmse_symbol_errors=g['mmse_symbol_errors'],candidate_BER=g['candidate_ber'],MMSE_BER=g['mmse_ber'],
                         candidate_SER=g['candidate_ser'],MMSE_SER=g['mmse_ser'],candidate_raw_EVM=g['mean_candidate_raw_evm'],MMSE_raw_EVM=g['mean_mmse_raw_evm'],
                         joint_outages=g['outage_bounds']['joint']['count'],joint_upper=g['outage_bounds']['joint']['upper'],status=g['status'],source_cycles=g['source_finish_cycle']))
        for s in g['stream_metrics']:streams.append(dict(operator=g['operator'],es_no_db=g['es_no_db'],**s))
    table(output/'quality.csv',rows);table(output/'per_stream.csv',streams)
    table(output/'constellation_mapping.csv',[dict(bit_I=i,bit_Q=q,I=(2*i-1)/math.sqrt(2),Q=(2*q-1)/math.sqrt(2)) for i in (0,1) for q in (0,1)])
    source=json.loads(Path(c['source_index']['path']).read_text());pipelines=[]
    for item in source['cases']:
        p=Path(item['result_path']).parent/'pipeline_statistics.json';stats=json.loads(p.read_text())
        for unit,d in stats['pipelines'].items():
            sum_core_union=sum(core['units'].get(unit,{}).get('active_union',0.) for core in stats['cores'].values())
            pipelines.append(dict(operator=item['operator'],noise_variance=item['regularization'],unit=unit,source_cycles=stats['finish_cycle'],
                                  events=d['event_count'],duration_sum=d['duration_sum'],global_active_union=d['active_union'],sum_core_active_union=sum_core_union,
                                  average_core_event_coverage=sum_core_union/(24*stats['finish_cycle']),mean_duration=d['mean_duration'],max_duration=d['max_duration']))
    table(output/'pipeline_cycles.csv',pipelines)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    operators=list(dict.fromkeys(g['operator'] for g in r['groups']));levels=sorted({g['es_no_db'] for g in r['groups']})
    for metric,label in [('ber','BER'),('ser','SER')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for op in operators:
            gg=sorted([g for g in r['groups'] if g['operator']==op],key=lambda g:g['es_no_db'])
            denom='bits' if metric=='ber' else 'symbols'
            ax.semilogy([g['es_no_db'] for g in gg],[g['candidate_'+metric] if g['candidate_'+metric] else 1/g[denom] for g in gg],marker='o',label=op)
        gg=sorted([g for g in r['groups'] if g['operator']==operators[0]],key=lambda g:g['es_no_db'])
        denom='bits' if metric=='ber' else 'symbols'
        ax.semilogy([g['es_no_db'] for g in gg],[g['mmse_'+metric] if g['mmse_'+metric] else 1/g[denom] for g in gg],'k--',label='FP64 MMSE')
        ax.set(xlabel='Es/N0 per unit-energy stream (dB)',ylabel=label,title='Uncoded QPSK / Rayleigh / actual instruction replay')
        ax.set_xticks(levels);ax.grid(True,which='both',alpha=.3);ax.legend(fontsize=7)
        fig.text(.01,.01,'Zero counts shown at 1/N for display only; this is not a confidence bound.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/(label+'.png'),dpi=220);fig.savefig(output/(label+'.pdf'));plt.close(fig)
    control=json.loads((root/'modem_controls.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['ber','ser']):
        ax.semilogy([a['es_no_db'] for a in control['awgn']],[a[metric] for a in control['awgn']],'o',label='Monte Carlo')
        ax.semilogy([a['es_no_db'] for a in control['awgn']],[a['theory_'+metric] for a in control['awgn']],'-',label='Analytic AWGN')
        ax.set(xlabel='Es/N0 (dB)',ylabel=metric.upper(),title='Modem/noise control, not MIMO reference');ax.grid(True);ax.legend()
    fig.tight_layout();fig.savefig(output/'AWGN_controls.png',dpi=220);fig.savefig(output/'AWGN_controls.pdf');plt.close(fig)
    fig,axes=plt.subplots(len(operators),len(levels),figsize=(9,2.2*len(operators)),squeeze=False)
    for i,op in enumerate(operators):
        for j,db in enumerate(levels):
            f=next(f for f in r['frames'] if f['operator']==op and abs(-10*math.log10(f['regularization'])-db)<1e-9)
            with np.load(root/f['output_artifact']['path']) as n:raw=n['actual_output_block'].astype(float)
            U=raw.shape[1]//2;R=raw.shape[2]//2;z=(raw[:,:U,:R]+1j*raw[:,U:,:R]).reshape(-1)[:2000]
            ax=axes[i,j];ax.scatter(z.real,z.imag,s=2,alpha=.15);ax.set_xlim(-1.8,1.8);ax.set_ylim(-1.8,1.8);ax.set_aspect('equal');ax.grid(True,alpha=.2)
            ideal=(np.array([-1,-1,1,1])+1j*np.array([-1,1,-1,1]))/math.sqrt(2)
            ax.scatter(ideal.real,ideal.imag,marker='x',c='red',s=20)
            ax.set_title(op+' / '+str(int(db))+' dB',fontsize=7)
    fig.suptitle('Actual soft outputs (subset, axes +/-1.8); red crosses: ideal QPSK; no gain fitting',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/'constellations.png',dpi=180);fig.savefig(output/'constellations.pdf');plt.close(fig)
    if sha(path)!=expected:raise ValueError('source result changed during report')
    manifest=dict(schema='qpsk-report-v1',source_result=str(path),source_sha256=expected,audit=checks,
                  matplotlib=matplotlib.__version__,reporter_sha256=sha(__file__),
                  artifacts=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--result',required=True,type=Path);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();report=run(a.result,a.sha256,a.output);print(json.dumps(report['audit']))
