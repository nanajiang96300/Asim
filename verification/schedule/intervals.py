"""Sparse byte-interval writer maps for fragmented DMA hazard checking."""
from bisect import bisect_left,bisect_right
from verification.schedule.hazards import validate_schedule_events

class Writers:
    def __init__(self):self.parts=[];self.starts=[]
    def query(self,lo,hi):
        i=max(0,bisect_right(self.starts,lo)-1)
        while i<len(self.parts):
            a,b,writer=self.parts[i]
            if a>=hi:break
            if b>lo:yield max(a,lo),min(b,hi),writer
            i+=1
    def assign(self,lo,hi,writer):
        left=bisect_left(self.starts,lo)
        if left and self.parts[left-1][1]>lo:left-=1
        right=bisect_left(self.starts,hi);new=[]
        if left<right and self.parts[left][0]<lo:new.append((self.parts[left][0],lo,self.parts[left][2]))
        new.append((lo,hi,writer))
        if left<right and self.parts[right-1][1]>hi:new.append((hi,self.parts[right-1][1],self.parts[right-1][2]))
        self.parts[left:right]=new;self.starts[left:right]=[p[0] for p in new]


def check_schedule_fast(operations,events,initial=()):
    seen=validate_schedule_events(operations,events,initial);faults=[];writers={}
    for a in initial:writers.setdefault(a.key(),Writers()).assign(a.lo,a.hi,None)
    for o in operations:
        name=o['id']
        for a in o['reads']:
            end=a.lo
            for lo,hi,w in writers.setdefault(a.key(),Writers()).query(a.lo,a.hi):
                if lo>end:faults.append({'hazard':'UNINITIALIZED','reader':name})
                end=hi
                if w is not None and seen[w,'write_visible']>seen[name,'read_begin']:faults.append({'hazard':'RAW','writer':w,'reader':name})
            if end<a.hi:faults.append({'hazard':'UNINITIALIZED','reader':name})
        for a in o['writes']:
            index=writers.setdefault(a.key(),Writers())
            for _,_,w in index.query(a.lo,a.hi):
                if w is not None and seen[w,'write_visible']>seen[name,'write_visible']:faults.append({'hazard':'WAW','earlier_writer':w,'later_writer':name})
            index.assign(a.lo,a.hi,name)
    # Nearest following writer per byte suffices; any out-of-order later writer
    # is rejected by WAW above, even if it is hidden behind this nearest writer.
    writers={}
    for o in reversed(operations):
        name=o['id']
        for a in o['reads']:
            for _,_,w in writers.setdefault(a.key(),Writers()).query(a.lo,a.hi):
                if seen[w,'write_visible']<seen[name,'read_done']:faults.append({'hazard':'WAR','reader':name,'writer':w})
        for a in o['writes']:writers.setdefault(a.key(),Writers()).assign(a.lo,a.hi,name)
    return {'status':'SCHEDULE_MODEL_REJECTED' if faults else 'SCHEDULE_MODEL_CHECKED','faults':faults[:64],'fault_count':len(faults)}
