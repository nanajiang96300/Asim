from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/nanajiang/Asim/experiments/optimization/bri/inner-0001');d=json.loads((p/'summary.json').read_text())['cases'];x=np.arange(3);fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,keys,label in [(axes[0],['baseline_cycles','candidate_cycles'],'Actual Asim cycles (batch96, 24 cores)'),(axes[1],['baseline_instruction_count','candidate_instruction_count'],'Actual source instruction count (all96 batches)')]:
 for j,key in enumerate(keys):
  bars=ax.bar(x+(j-.5)*.36,[r[key] for r in d],.36,label='baseline' if j==0 else 'inner-0001')
  ax.bar_label(bars,labels=[f'{r[key]:,}' for r in d],fontsize=7,padding=3)
 ax.set_xticks(x,[f"{r['snr_db']}dB / K{r['iterations']}" for r in d]);ax.set_ylabel(label);ax.set_ylim(0,max(r[keys[0]] for r in d)*1.2);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('BRI fixed-HB state rotation: same FP16 output bits, same task and K',fontsize=11);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'cycle_instruction_comparison.{ext}',dpi=180)
