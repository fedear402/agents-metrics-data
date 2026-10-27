"""Use the existing queue with the unlimited-walltime collector entry point."""
from pathlib import Path
import production_queue as central_queue

central_queue.COLLECTOR=Path(__file__).with_name('prospective_collect_unlimited.py')

if __name__=='__main__':
    central_queue.main()
