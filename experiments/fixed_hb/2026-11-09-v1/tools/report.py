from pathlib import Path
import json,csv,shutil,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path('/home/nanajiang/Asim');raw=repo/'results/fixed_hb/2026-11-09-v1';out=repo/'experiments/fixed_hb/2026-11-09-v1'
a=json.loads((raw/'calibration/result.json').read_text());b=json.loads((raw/'holdout/result.json').read_text());sel=json.loads((raw/'selection.json').read_text())
assert a['status']=='CALIBRATION_SELECTED' and b['status']=='FIXED_HB_NUMERIC_HOLDOUT_CHECKED'
fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True);rows=[]
for ax,cal,hold,s in zip(axes,a['cases'],b['cases'],sel['cases']):
 for group,case,style in [('calibration',cal,'-'),('fresh algorithm holdout',hold,'--')]:
  curve=case['curves'];ax.semilogy([x['K'] for x in curve],[100*x['max_error'] for x in curve],style,label=group)
  rows.extend(dict(domain=group,regularization=case['lambda_'],**r) for r in curve)
 ax.axhline(2,color='red',ls=':',label='acceptance 2%');ax.axhline(1,color='gray',ls=':',label='calibration margin 1%');ax.axvline(s['K'],color='green',ls='-.',label=f"frozen K={s['K']}")
 ax.set_title(f"SNR {int(-10*__import__('math').log10(s['lambda_']))} dB");ax.set_xlabel('Iterations K');ax.grid(True,alpha=.25)
axes[0].set_ylabel('Max relative MMSE output error (%)');axes[-1].legend(fontsize=7);fig.suptitle('Fixed-HB FP16 storage / FP32 temporary model; algorithm screening only');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'iterations_error.{ext}',dpi=180)
with (out/'iterations_error.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
for src,name in [(repo/'results/fixed-hb-math-review-20261109.md','math_review.md'),(repo/'results/fixed-hb-unit-tests-20261109-final.log','unit-tests.log'),(raw/'old_failure_regression.json','old_failure_regression.json')]:shutil.copy2(src,out/name)
summary=dict(status=b['status'],selection=sel,calibration=[{k:v for k,v in c.items() if k!='curves'} for c in a['cases']],holdout=[{k:v for k,v in c.items() if k!='curves'} for c in b['cases']],pure_model_thinking_seconds=None,scope='algorithm screening, not actual instruction reconstruction or hardware certification')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(status=summary['status'],curves=len(rows))))
