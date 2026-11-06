#!/usr/bin/env python3
"""Run a trusted command and record elapsed tool time, including failures."""
import argparse,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import fcntl

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ledger',type=Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:ap.error('command required')
    start=time.time_ns();mono=time.monotonic_ns();rc=127
    try:rc=subprocess.run(cmd).returncode;return rc
    finally:
        elapsed=time.monotonic_ns()-mono
        e=dict(schema='tool-span-v1',label=a.label,start_ns=start,end_ns=start+elapsed,elapsed_seconds=elapsed/1e9,returncode=rc,command=cmd)
        a.ledger.parent.mkdir(parents=True,exist_ok=True)
        with a.ledger.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(e)+'\n');f.flush()
if __name__=='__main__':raise SystemExit(main())
