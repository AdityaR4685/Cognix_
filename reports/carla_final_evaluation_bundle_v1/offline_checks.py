"""Run only allowlisted verification under a network/subprocess refusal hook."""
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
allowed={'verify_bundle.py','offline_regression.py','verify_results.py'}
target=Path(sys.argv[1]).resolve();args=sys.argv[2:]
if target.parent!=HERE or target.name not in allowed:raise RuntimeError('OFFLINE_CHECK_ONLY_NO_EXECUTION')
attempts=[]
def guard(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.sendto','socket.sendmsg','urllib.Request','http.client.connect','subprocess.Popen','os.system','os.posix_spawn','os.spawn','os.exec'}:
        attempts.append(event);raise RuntimeError('OFFLINE_NETWORK_OR_PROCESS_ACCESS_FORBIDDEN: '+event)
sys.addaudithook(guard)
sys.path.insert(0,str(HERE));sys.argv=[str(target),*args]
runpy.run_path(str(target),run_name='__main__')
if attempts:raise RuntimeError('OFFLINE_NETWORK_ATTEMPT_DETECTED')
print(json.dumps({'offline_network_guard':'PASSED','actual_network_calls':0,'TEST_network_requests':0,'TEST_payload_bytes_accessed':0}))
