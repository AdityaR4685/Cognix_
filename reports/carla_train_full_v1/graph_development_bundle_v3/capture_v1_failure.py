"""Read-only reproduction below v1 CLI publication: never reuses/resumes its pending workspace."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
old=Path(__file__).resolve().parent.with_name('graph_development_bundle_v1')
sys.path.insert(0,str(old))
import graph_common as v1
v1.verify_seal(old,'7da8ffb4ded5ee446f07b369577a9f62f5191592c9da682f680ac76705fb8ece')
v1.install_guard('preflight')  # Actual audit boundary is read-only and denies archive payload opens.
import graph_replay
def stop_scientific_execution(frame,event,arg):
    if event=='call' and frame.f_code.co_name in ('restore_current','generate_pair','generate_pseudo_anomaly','feed','fit','fit_calibrator'):
        raise RuntimeError('Read-only failure probe cannot proceed into restoration, replay, pseudo or fitting')
sys.setprofile(stop_scientific_execution)
# This phase marker admits the existing replay_current entry assertion only;
# the installed v1 audit hook remains PRELIGHT/read-only throughout the probe.
# The unedited v1 verifier then fails before this function reaches any science.
v1.ACTIVE_PHASE='graph-data'
graph_replay.replay_current()
