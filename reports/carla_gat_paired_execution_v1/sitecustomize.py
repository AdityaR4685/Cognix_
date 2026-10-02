"""Observe only the frozen trainer child; do not alter its implementation."""
import os
import sys

if os.environ.get('COGNIX_FULL_OBSERVER') == '1' and sys.argv[0].endswith('/carla_graph_trainers_v1/train.py'):
    try:
        import execution_observer
        execution_observer.install()
    except BaseException:
        import traceback
        traceback.print_exc()
        # Python otherwise swallows sitecustomize exceptions and continues.
        os._exit(91)
