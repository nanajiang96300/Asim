"""Trace diagnostics: summed residency is not end-to-end or PE utilization."""
from collections import Counter,defaultdict
import csv
import math
import re

OPCODES=['MOVIN','MOVOUT','MOVOUT_POOL','GEMM_PRELOAD','GEMM','GEMM_WRITE','COMP','IM2COL','SOFTMAX','LAYERNORM','ADD','MUL','MAC','DIV','ADDTREE','EXP','SQRT','SCALAR_ADD','SCALAR_SUB','SCALAR_MUL','SCALAR_DIV','SCALAR_SQRT','GELU','SWISH','BAR','PIPE_BARRIER']


def union(intervals):
    total=0;end=None
    for start,stop in sorted(intervals):
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total


def metrics(rows):
    durations=[end-start for start,end in rows]
    return {'event_count':len(rows),'duration_sum':sum(durations),'active_union':union(rows),
            'min_duration':min(durations) if rows else None,'max_duration':max(durations) if rows else None,
            'mean_duration':sum(durations)/len(rows) if rows else None,
            'first_start':min((x[0] for x in rows),default=None),'last_end':max((x[1] for x in rows),default=None)}


def statistics(trace_path,instructions,finish_cycle,num_cores):
    opcode_counts=Counter(OPCODES[i['opcode']] for i in instructions)
    names=defaultdict(set);barriers=set()
    for i in instructions:
        if i['id']:names[i['id']].add(OPCODES[i['opcode']])
        if i['barrier']:barriers.add(i['id'])
    per_unit=defaultdict(list);per_core=defaultdict(lambda:defaultdict(list));detail=defaultdict(list);errors=[]
    with open(trace_path) as stream:
        for row in csv.DictReader(stream):
            try:
                start=float(row['start_cycle']);end=float(row['end_cycle'])
                if not math.isfinite(start) or not math.isfinite(end) or start<0 or end<start or (finish_cycle is not None and end>finish_cycle):raise ValueError('invalid interval')
                match=re.fullmatch(r'Core(\d+)_(Cube|Vector|Scalar|MTE2|MTE3|Wait)',row['unit'])
                if not match:raise ValueError('unknown pipeline unit')
                core=int(match[1]);unit=match[2]
                if core>=num_cores:raise ValueError('core outside config')
                kind='Barrier' if row['name'] in barriers and unit=='Vector' else unit
                per_unit[kind].append((start,end));per_core[core][kind].append((start,end))
                possible=names.get(row['name'],set())
                opcode=next(iter(possible)) if len(possible)==1 else 'DMA_FRAGMENT' if unit in ('MTE2','MTE3') else 'WAIT' if unit=='Wait' else 'AMBIGUOUS'
                detail[(core,unit,opcode,row['name'])].append((start,end))
            except (ValueError,KeyError) as error:errors.append(str(error))
    rows=[{'core':key[0],'unit':key[1],'opcode':key[2],'name':key[3],**metrics(value)} for key,value in sorted(detail.items())]
    cores={}
    for core in range(num_cores):
        units=per_core[core];busy=[span for kind,spans in units.items() if kind!='Wait' for span in spans];active=union(busy)
        changes=defaultdict(Counter)
        for kind,spans in units.items():
            if kind=='Wait':continue
            for start,end in spans:changes[start][kind]+=1;changes[end][kind]-=1
        state=Counter();previous=0;overlap=0
        for t,deltas in sorted(changes.items()):
            if sum(n>0 for n in state.values())>=2:overlap+=t-previous
            state.update(deltas);previous=t
        cores[str(core)]={'units':{k:metrics(v) for k,v in units.items()},'busy_union':active,'idle_in_makespan':finish_cycle-active if finish_cycle is not None else None,'multi_unit_overlap_union':overlap}
    return {'status':'TRACE_VALID' if not errors else 'TRACE_INVALID','errors':errors,'finish_cycle':finish_cycle,
            'source_instruction_count':len(instructions),'source_opcode_counts':dict(opcode_counts),
            'pipelines':{k:metrics(v) for k,v in per_unit.items()},'cores':cores,'instruction_pipeline_detail':rows,
            'limitations':['duration_sum may overlap across instructions/cores; never sum it into makespan','MTE events are request fragments, not unique source instructions','Wait is observed CubeWait only, not complete stall attribution','legacy IDs repeat; these statistics are not G4 event identity proof','MTE1 not separately instrumented; no invented cycles']}


def write_detail(path,stats):
    fields=['core','unit','opcode','name','event_count','duration_sum','active_union','min_duration','max_duration','mean_duration','first_start','last_end']
    with open(path,'w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(stats['instruction_pipeline_detail'])
