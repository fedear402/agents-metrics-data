"""Supervise the unlimited queue that records additive R packages."""
from pathlib import Path
import shlex
import collection_supervisor_unlimited as base
central_supervisor=base.central_supervisor
central_supervisor.queue.COLLECTOR=Path(__file__).with_name('prospective_collect_unlimited_additions.py')
BASE_LIVE=central_supervisor.live_controller
def live_controller(entry,processes):
    normalized=[]
    for pid,command in processes:
        try:args=shlex.split(command)
        except ValueError:continue
        args=[str(Path(arg).with_name('production_queue_unlimited.py')) if Path(arg).name=='production_queue_unlimited_additions.py' else arg for arg in args]
        normalized.append((pid,shlex.join(args)))
    return BASE_LIVE(entry,normalized)
central_supervisor.live_controller=live_controller
if __name__=='__main__':central_supervisor.main()
