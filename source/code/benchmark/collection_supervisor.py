"""Monitor authorized queue commands and restart eligible work after a queue exits.

This program never changes run records, manifests, prompts, or provider settings.
It checks every chosen interval. Existing controllers and active collection locks
prevent another launch. Held or ambiguous requests remain excluded by the queue.
"""
from __future__ import annotations
import argparse,json,os,shlex,subprocess,time,fcntl
from collections import Counter
from pathlib import Path
import production_queue as queue


def live_controller(entry, processes):
    """Recognize either the uv parent or Python queue process by scoped arguments."""
    for pid,command in processes:
        try:args=shlex.split(command)
        except ValueError:continue
        if not any(Path(a).name=='production_queue.py' for a in args):continue
        if '--index' not in args:continue
        i=args.index('--index')
        if i+1>=len(args) or str(Path(args[i+1]).resolve())!=str(Path(entry['index']).resolve()):continue
        if all(batch in args for batch in entry['batches']):return pid
    return None


def process_snapshot():
    result=subprocess.run(['ps','-axo','pid=,command='],check=True,text=True,capture_output=True)
    return [(int(parts[0]),parts[1]) for line in result.stdout.splitlines()
            if len(parts:=line.strip().split(None,1))==2]


def inspect(entry, processes):
    counts=Counter();pending=0;active=[]
    selected=[x for x in queue.checked_index(Path(entry['index'])) if x['batch'] in entry['batches']]
    if {x['batch'] for x in selected}!=set(entry['batches']):raise ValueError('Queue batch scope changed')
    for batch in selected:
        path=Path(batch['manifest']);snapshot=queue.status(path)
        counts.update(row['status'] for row in snapshot['rows'])
        jobs,_=queue.candidates(path,snapshot);pending+=len(jobs)
        if queue.collection_active(path):active.append(batch['batch'])
    pid=live_controller(entry,processes)
    return dict(counts=dict(counts),eligible_pending=pending,active_batches=active,controller_pid=pid,
                restart_eligible=bool(pending and not active and pid is None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-file',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=60)
    parser.add_argument('--restart-delay',type=float,default=300)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    children={};last_launch={}
    with (args.out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (args.out/'STOP').exists():
            # Reap owned children. Existing queues remain independent.
            for name,child in list(children.items()):
                if child.poll() is not None:del children[name]
            report=dict(at=queue.timestamp(),supervisor_pid=os.getpid(),interval_seconds=args.interval,
                        restart_delay_seconds=args.restart_delay,queues={})
            try:processes=process_snapshot()
            except Exception as error:
                report['process_inspection_error']=str(error);processes=None
            for source in args.launch_file:
                if not source.exists():continue
                for entry in queue.read(source):
                    name=str(source.resolve())+'::'+entry['name']
                    try:
                        if processes is None:raise RuntimeError('Process inspection unavailable; no restart attempted')
                        state=inspect(entry,processes)
                        if (state['restart_eligible'] and name not in children
                                and time.monotonic()-last_launch.get(name,-args.restart_delay)>=args.restart_delay):
                            env=os.environ.copy();env.update(entry.get('env',{}))
                            with Path(entry['log']).open('a') as log:
                                child=subprocess.Popen(entry['argv'],cwd=entry['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            children[name]=child;last_launch[name]=time.monotonic();state['launched_pid']=child.pid
                            with (args.out/'restarts.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=queue.timestamp(),name=name,pid=child.pid,eligible_pending=state['eligible_pending']))+'\n')
                        report['queues'][name]=state
                    except Exception as error:report['queues'][name]=dict(error=str(error),restart_attempted=False)
            queue.save(args.out/'status.json',report)
            if args.once:break
            time.sleep(args.interval)

if __name__=='__main__':main()
