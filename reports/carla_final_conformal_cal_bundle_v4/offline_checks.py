"""Run only offline verification under a process-local network refusal hook."""
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
allowed = {"verify_bundle.py", "synthetic_verification.py", "regression_verification.py", "oov_regression_verification.py", "fresh_split_verification.py", "runner.py"}
if sys.argv[1:] == ['--guard-selftest']:
    target, arguments = None, []
else:
    target = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
if target is not None and (target.parent != HERE or target.name not in allowed or (target.name == "runner.py" and arguments != ["--preflight"])):
    raise RuntimeError("OFFLINE_CHECK_ONLY_NO_SCIENTIFIC_EXECUTION")
network_attempts = []
def deny_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg", "urllib.Request", "http.client.connect", "subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn", "os.exec"}:
        network_attempts.append(event)
        raise RuntimeError("OFFLINE_NETWORK_ACCESS_FORBIDDEN: " + event)
sys.addaudithook(deny_network)
if target is None:
    # Inject synthetic audit events only; no actual networking or subprocess API call.
    for event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto','socket.sendmsg','urllib.Request','http.client.connect','subprocess.Popen','os.system','os.posix_spawn','os.spawn','os.exec'):
        try: sys.audit(event, None)
        except RuntimeError as exc: assert str(exc).startswith('OFFLINE_NETWORK_ACCESS_FORBIDDEN:')
        else: raise AssertionError('Guard failed: '+event)
    print(json.dumps({'status':'PASSED','synthetic_audit_events_refused':len(network_attempts),'actual_network_calls':0}))
    network_attempts.clear()
else:
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name="__main__")
if network_attempts:
    raise RuntimeError("OFFLINE_NETWORK_ATTEMPT_DETECTED")
print(json.dumps({"offline_network_guard": "PASSED", "network_attempts": 0,
                  "TEST_network_requests": 0, "TEST_payload_bytes_accessed": 0}))
