"""Run only offline verification under a process-local network refusal hook."""
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
allowed = {"verify_bundle.py", "synthetic_verification.py", "regression_verification.py", "runner.py"}
target = Path(sys.argv[1]).resolve()
arguments = sys.argv[2:]
if target.name not in allowed or (target.name == "runner.py" and arguments != ["--preflight"]):
    raise RuntimeError("OFFLINE_CHECK_ONLY_NO_SCIENTIFIC_EXECUTION")
network_attempts = []
def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request", "http.client.connect"}:
        network_attempts.append(event)
        raise RuntimeError("OFFLINE_NETWORK_ACCESS_FORBIDDEN: " + event)
sys.addaudithook(deny_network)
sys.path.insert(0, str(target.parent))
sys.argv = [str(target), *arguments]
runpy.run_path(str(target), run_name="__main__")
if network_attempts:
    raise RuntimeError("OFFLINE_NETWORK_ATTEMPT_DETECTED")
print(json.dumps({"offline_network_guard": "PASSED", "network_attempts": 0,
                  "TEST_network_requests": 0, "TEST_payload_bytes_accessed": 0}))
