"""Run the central queue with recorded additive R package changes."""
from pathlib import Path
import production_queue as central_queue
central_queue.COLLECTOR=Path(__file__).with_name('prospective_collect_unlimited_additions.py')
if __name__=='__main__':central_queue.main()
