"""Read/write event contract and conservative RAW/WAR/WAW checker.

Synthetic model tests only until Core emits bound read-complete/write-visible events.
A pipeline lifetime CSV is not this event stream and cannot certify G4.
`order` is a unique caller-supplied tie breaker within a cycle, not an
authenticated global clock or emission order. Runtime clock binding is pending.
"""
from dataclasses import dataclass
from pathlib import Path
from verification.contracts.core import Invalid,Unsupported

@dataclass(frozen=True)
class Access:
    space:str
    core:int
    buffer:int
    lo:int
    hi:int
    def key(self):return self.space,self.core,self.buffer
    def validate(self):
        if self.space not in ('DRAM','SPAD','ACCUM') or any(type(v) is not int for v in (self.core,self.buffer,self.lo,self.hi)) or not 0<=self.lo<self.hi:raise Invalid('invalid physical access')
        if self.space=='DRAM':
            if (self.core,self.buffer)!=(-1,-1):raise Invalid('DRAM must use global address identity')
        elif self.core<0 or self.buffer not in (0,1):raise Invalid('invalid local core/buffer')

def overlap(a,b):return a.key()==b.key() and max(a.lo,b.lo)<min(a.hi,b.hi)

def subtract(ranges,lo,hi):
    result=[]
    for a,b in ranges:
        if b<=lo or hi<=a:result.append((a,b));continue
        if a<lo:result.append((a,lo))
        if hi<b:result.append((hi,b))
    return result


def validate_schedule_events(operations,events,initial=()):
    ids=[o['id'] for o in operations]
    if not ids or len(set(ids))!=len(ids):raise Invalid('empty/duplicate op identities')
    by_id={o['id']:o for o in operations};seen={};stamps=set()
    for op in operations:
        for a in op['reads']+op['writes']:a.validate()
    for a in initial:a.validate()
    for e in events:
        name,kind=e['op'],e['kind'];cycle,order=e['cycle'],e['order']
        if name not in by_id or kind not in ('read_begin','read_done','write_visible') or any(type(x) is not int or x<0 for x in (cycle,order)):raise Invalid('invalid runtime event')
        if (name,kind) in seen or order in stamps:raise Invalid('duplicate event or ambiguous event ordering')
        stamps.add(order);seen[(name,kind)]=(cycle,order)
    coverage={name:set() for name in ids}
    for name,kind in seen:coverage[name].add(kind)
    for o in operations:
        required=({'read_begin','read_done'} if o['reads'] else set())|({'write_visible'} if o['writes'] else set())
        if coverage[o['id']]!=required:raise Unsupported('missing/extra read completion or visibility events')
        if o['reads'] and seen[o['id'],'read_done']<seen[o['id'],'read_begin']:raise Invalid('read completes before beginning')
        if o['reads'] and o['writes'] and seen[o['id'],'write_visible']<seen[o['id'],'read_done']:raise Invalid('unsupported streaming write before complete read')
    return seen


def check_schedule(operations,events,initial=()):
    seen=validate_schedule_events(operations,events,initial)
    faults=[]
    for index,o in enumerate(operations):
        name=o['id']
        for read in o['reads']:
            uncovered=[(read.lo,read.hi)]
            # Last intended writer per byte wins, including partial overwrites.
            for prior in reversed(operations[:index]):
                for write in prior['writes']:
                    if not overlap(read,write) or not any(max(a,write.lo)<min(b,write.hi) for a,b in uncovered):continue
                    if seen[prior['id'],'write_visible']>seen[name,'read_begin']:faults.append({'hazard':'RAW','writer':prior['id'],'reader':name})
                    uncovered=subtract(uncovered,write.lo,write.hi)
            for a in initial:
                if a.key()==read.key():uncovered=subtract(uncovered,a.lo,a.hi)
            if uncovered:faults.append({'hazard':'UNINITIALIZED','reader':name,'ranges':uncovered})
            for later in operations[index+1:]:
                if any(overlap(read,w) for w in later['writes']) and seen[later['id'],'write_visible']<seen[name,'read_done']:
                    faults.append({'hazard':'WAR','reader':name,'writer':later['id']})
        for write in o['writes']:
            for prior in operations[:index]:
                if any(overlap(write,w) for w in prior['writes']) and seen[prior['id'],'write_visible']>seen[name,'write_visible']:
                    faults.append({'hazard':'WAW','earlier_writer':prior['id'],'later_writer':name})
    return {'status':'SCHEDULE_MODEL_REJECTED' if faults else 'SCHEDULE_MODEL_CHECKED','faults':faults,'scope':'event-model conformance only; real G4 needs trusted Core read/write/DMA bindings'}


def require_runtime_rw(directory):
    path=Path(directory)/'rw_events.jsonl'
    if not path.is_file():raise Unsupported('G4_MISSING_READ_COMPLETION_AND_WRITE_VISIBILITY: pipeline trace is insufficient')
    raise Unsupported('G4_CORE_EVENT_BINDING_NOT_IMPLEMENTED: do not accept an unbound event file')
