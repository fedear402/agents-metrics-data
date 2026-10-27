"""Run the central queue with the usage-watched collector entry point."""
from pathlib import Path
import production_queue as central_queue

central_queue.COLLECTOR = Path(__file__).with_name('prospective_collect_watched.py')

if __name__ == '__main__':
    central_queue.main()
