"""Supervise unlimited-walltime queues with the existing lock and hold rules."""
from pathlib import Path
import shlex
import collection_supervisor as central_supervisor

central_supervisor.queue.COLLECTOR=Path(__file__).with_name('prospective_collect_unlimited.py')
BASE_LIVE_CONTROLLER=central_supervisor.live_controller


def live_controller(entry,processes):
    """Recognize this entry point using the central index and batch checks."""
    normalized=[]
    for pid,command in processes:
        try:
            args=shlex.split(command)
        except ValueError:
            continue
        args=[str(Path(arg).with_name('production_queue.py'))
              if Path(arg).name=='production_queue_unlimited.py' else arg for arg in args]
        normalized.append((pid,shlex.join(args)))
    return BASE_LIVE_CONTROLLER(entry,normalized)


central_supervisor.live_controller=live_controller

if __name__=='__main__':
    central_supervisor.main()
