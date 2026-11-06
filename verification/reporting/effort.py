"""Observable wall-time accounting. Never claims access to model reasoning time."""
from datetime import datetime,timezone

def summarize(start_ns,end_ns,spans):
    if end_ns<start_ns:raise ValueError('invalid session interval')
    intervals=[]
    for e in spans:
        a,b=e['start_ns'],e['end_ns']
        if b<a:raise ValueError('invalid tool interval')
        a=max(start_ns,a);b=min(end_ns,b)
        if b>a:intervals.append((a,b))
    union=0;left=right=None
    for a,b in sorted(intervals):
        if right is None:left,right=a,b
        elif a<=right:right=max(right,b)
        else:union+=right-left;left,right=a,b
    if right is not None:union+=right-left
    total=(end_ns-start_ns)/1e9
    return dict(schema='agent-effort-v1',session_wall_seconds=total,observed_tool_union_seconds=union/1e9,
                agent_or_uninstrumented_seconds=total-union/1e9,pure_model_thinking_seconds=None,
                definition='residual includes reasoning, writing, scheduling, uninstrumented tools and waits; not pure thinking',
                start_utc=datetime.fromtimestamp(start_ns/1e9,timezone.utc).isoformat(),end_utc=datetime.fromtimestamp(end_ns/1e9,timezone.utc).isoformat(),
                command_spans=spans)
