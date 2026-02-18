import multiprocessing
import sys

from core.cli import main

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    sys.exit(main())
