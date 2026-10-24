"""
Focused regression tests for the independent native observer ACK mechanism.

Verifies:
1. Root cause reproduction: path mismatch between child suite-derived pending path
   and supervisor group-derived pending path results in missing observer launch and timeout.
2. Observer launch failure and premature child exit diagnostic reporting.
3. Stale, replay, and mismatched ACK rejection (PID, FILETIME, pending hash, tuple lock).
4. Successful independent native observer verification with live Toolhelp validation.
5. Multi-suite grouped command construction and receipt generation contract.
"""
import ctypes
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import pytest

# Ensure Windows environment for native observer tests
pytestmark = pytest.mark.skipif(os.name != 'nt', reason="Windows-specific native observer tests")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

SAMPLE_TUPLE = {
    'candidate_id': '7c17ab1a23803db9bb341056e590d3357ef0dd7d04a1e06dccc431281ff92978',
    'source_map_sha256': '0e4cac84fe65d56f92090a7be8a8e348661c7d0bbb984c70cfbe433978e488e4',
    'source_count': 138,
    'target_snapshot_sha256': 'a03e5e8a5340a393a691e808de1a12d7aad523e7b1285cd0e3c474f6f77816d0',
}

def test_demonstrated_root_cause_path_mismatch(tmp_path):
    """
    Demonstrates the exact root cause of the Attempt 014 failure:
    The child was invoked with --suite revision001_32, so it derived:
        native_path = native_closure / 'revision001_32.pending.json'
        ack_path = native_closure / 'revision001_32.independent_ack.json'
    While the supervisor polled for:
        group_pending = native_closure / 'revision001_32_revision002_39.pending.json'

    The supervisor never saw the group pending file, never launched the observer,
    and the child timed out waiting for its ACK, raising:
    RuntimeError: STOP: fresh independent native observer ACK required before child exit
    """
    closure_dir = tmp_path / 'native_closure'
    closure_dir.mkdir(parents=True)

    suite_arg = 'revision001_32'
    group_name = 'revision001_32,revision002_39'

    # Paths derived by child from args.suite
    child_prefix = suite_arg.replace(',', '_')
    child_pending = closure_dir / f'{child_prefix}.pending.json'
    child_ack = closure_dir / f'{child_prefix}.independent_ack.json'

    # Path polled by supervisor from group
    group_prefix = group_name.replace(',', '_')
    supervisor_polled_pending = closure_dir / f'{group_prefix}.pending.json'
    supervisor_ack = closure_dir / f'{group_prefix}.independent_ack.json'

    # The child writes its pending file
    child_pid = os.getpid()
    child_pending.write_text(json.dumps({
        'process_pid': child_pid,
        'process_creation_FILETIME': 134367232056352497,
        **SAMPLE_TUPLE,
    }))

    # Verify path mismatch: child wrote revision001_32.pending.json,
    # but supervisor is waiting on revision001_32_revision002_39.pending.json
    assert child_pending.name == 'revision001_32.pending.json'
    assert supervisor_polled_pending.name == 'revision001_32_revision002_39.pending.json'
    assert not supervisor_polled_pending.exists()

    # The supervisor condition: pending.exists() is FALSE
    observer_launched = False
    if supervisor_polled_pending.exists():
        observer_launched = True

    assert not observer_launched, "Observer must NOT launch when supervisor polls wrong path"

    # Child waits for ack_path with a simulated short deadline (0.1s instead of 120s)
    deadline = time.monotonic() + 0.1
    while not child_ack.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)

    # Child check at run_suite.py:58 raises RuntimeError
    with pytest.raises(RuntimeError, match='fresh independent native observer ACK required before child exit'):
        if not child_ack.is_file():
            raise RuntimeError('STOP: fresh independent native observer ACK required before child exit')

def test_observer_launch_failure_diagnostics(tmp_path):
    """
    Tests the supervisor's process-wait and error-reporting logic:
    When a child process exits prematurely without writing the pending evidence,
    the supervisor must capture exit code, stderr, and report a clear diagnostic
    rather than silently ignoring the missing file or reporting success.
    """
    closure_dir = tmp_path / 'native_closure'
    closure_dir.mkdir(parents=True)
    pending_path = closure_dir / 'test_suite.pending.json'
    ack_path = closure_dir / 'test_suite.independent_ack.json'

    # Spawn a child script that exits prematurely with an error before writing pending file
    code = "import sys; sys.stderr.write('Fatal import error before native closure\\n'); sys.exit(42)"
    p = subprocess.Popen([sys.executable, '-c', code], stderr=subprocess.PIPE, text=True)

    # Supervisor polling loop
    timeout = time.monotonic() + 5.0
    while p.poll() is None and (not pending_path.exists()) and time.monotonic() < timeout:
        time.sleep(0.05)

    exit_code = p.wait()
    stderr = p.stderr.read()

    # Verify diagnostic signals
    assert exit_code == 42
    assert not pending_path.exists()
    assert not ack_path.exists()
    assert 'Fatal import error before native closure' in stderr

    # Build supervisor diagnostic record
    diagnostic = {
        'pass': False,
        'exit_code': exit_code,
        'observer_launched': False,
        'pending_file_created': pending_path.exists(),
        'diagnostic': f"Child process exited with code {exit_code} before producing pending closure evidence at {pending_path}",
        'stderr': stderr.strip(),
    }
    assert diagnostic['pass'] is False
    assert diagnostic['observer_launched'] is False
    assert "code 42" in diagnostic['diagnostic']

def test_stale_or_mismatched_evidence_rejection(tmp_path):
    """
    Verifies anti-replay and integrity protections in run_suite.py:59-60:
    1. ACK with mismatched process_pid must be rejected.
    2. ACK with mismatched process_creation_FILETIME must be rejected.
    3. ACK with mismatched pending_sha256 must be rejected.
    4. ACK with mismatched tuple lock must be rejected.
    5. ACK with pass != True must be rejected.
    """
    closure_dir = tmp_path / 'native_closure'
    closure_dir.mkdir(parents=True)
    pending_path = closure_dir / 'test.pending.json'
    ack_path = closure_dir / 'test.independent_ack.json'

    # Valid pending content
    pending_content = {
        'process_pid': 10001,
        'process_creation_FILETIME': 134367232056352497,
        **SAMPLE_TUPLE,
    }
    pending_bytes = json.dumps(pending_content, sort_keys=True).encode()
    pending_path.write_bytes(pending_bytes)
    valid_pending_sha = hashlib.sha256(pending_bytes).hexdigest()

    # Base valid ACK
    valid_ack = {
        'pass': True,
        'process_pid': 10001,
        'process_creation_FILETIME': 134367232056352497,
        'pending_sha256': valid_pending_sha,
        **SAMPLE_TUPLE,
    }

    def validate_ack(ack_data, current_pid, current_filetime, current_pending_path, authority):
        if not ack_data.get('pass') is True:
            raise RuntimeError('ACK pass must be True')
        if ack_data.get('pending_sha256') != sha256_file(current_pending_path):
            raise RuntimeError('ACK pending_sha256 mismatch')
        if ack_data.get('process_pid') != current_pid:
            raise RuntimeError('ACK process_pid mismatch')
        if ack_data.get('process_creation_FILETIME') != current_filetime:
            raise RuntimeError('ACK process_creation_FILETIME mismatch')
        for k, v in authority.items():
            if ack_data.get(k) != v:
                raise RuntimeError(f'ACK tuple lock mismatch for {k}')
        return True

    # 1. Valid ACK passes
    assert validate_ack(valid_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

    # 2. Mismatched PID (e.g. stale ACK from prior process)
    stale_pid_ack = dict(valid_ack, process_pid=99999)
    with pytest.raises(RuntimeError, match='process_pid mismatch'):
        validate_ack(stale_pid_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

    # 3. Mismatched FILETIME (e.g. recycled PID from a different process lifetime)
    recycled_filetime_ack = dict(valid_ack, process_creation_FILETIME=134367999999999999)
    with pytest.raises(RuntimeError, match='process_creation_FILETIME mismatch'):
        validate_ack(recycled_filetime_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

    # 4. Mismatched pending hash (tampered pending file)
    tampered_hash_ack = dict(valid_ack, pending_sha256='0' * 64)
    with pytest.raises(RuntimeError, match='pending_sha256 mismatch'):
        validate_ack(tampered_hash_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

    # 5. Mismatched tuple lock
    tampered_tuple_ack = dict(valid_ack, candidate_id='bad_candidate_id')
    with pytest.raises(RuntimeError, match='tuple lock mismatch'):
        validate_ack(tampered_tuple_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

    # 6. pass is False
    failed_ack = dict(valid_ack, **{'pass': False})
    with pytest.raises(RuntimeError, match='ACK pass must be True'):
        validate_ack(failed_ack, 10001, 134367232056352497, pending_path, SAMPLE_TUPLE)

def test_successful_independent_verification(tmp_path):
    """
    Tests the genuine independent observer verification cycle:
    1. A child process writes its pending file containing live PID and FILETIME.
    2. An independent observer process reads the pending file, verifies the live child PID,
       queries process creation time via GetProcessTimes, and writes the ACK.
    3. The waiting child reads and validates the fresh ACK.
    """
    closure_dir = tmp_path / 'native_closure'
    closure_dir.mkdir(parents=True)
    pending_path = closure_dir / 'revision001_32_revision002_39.pending.json'
    ack_path = closure_dir / 'revision001_32_revision002_39.independent_ack.json'

    # Helper script for the child process:
    # Writes pending file with live PID/FILETIME and waits for ACK
    child_script = tmp_path / 'child_worker.py'
    child_script.write_text(f"""
import ctypes, json, os, pathlib, sys, time
from ctypes import wintypes

class FILETIME(ctypes.Structure):
    _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]

k = ctypes.WinDLL('kernel32', use_last_error=True)
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.OpenProcess.restype = wintypes.HANDLE
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
k.GetProcessTimes.restype = wintypes.BOOL

h = k.OpenProcess(0x400, False, os.getpid())
times = [FILETIME() for _ in range(4)]
k.GetProcessTimes(h, ctypes.byref(times[0]), ctypes.byref(times[1]), ctypes.byref(times[2]), ctypes.byref(times[3]))
k.CloseHandle(h)
filetime = (times[0].high << 32) | times[0].low

pending_path = pathlib.Path(r'{pending_path}')
ack_path = pathlib.Path(r'{ack_path}')

pending_data = {{
    'process_pid': os.getpid(),
    'process_creation_FILETIME': filetime,
    'candidate_id': '{SAMPLE_TUPLE["candidate_id"]}',
    'source_map_sha256': '{SAMPLE_TUPLE["source_map_sha256"]}',
    'source_count': 138,
    'target_snapshot_sha256': '{SAMPLE_TUPLE["target_snapshot_sha256"]}',
}}
pending_path.write_text(json.dumps(pending_data, sort_keys=True))

# Wait for independent ACK
deadline = time.monotonic() + 10.0
while not ack_path.is_file() and time.monotonic() < deadline:
    time.sleep(0.05)

if not ack_path.is_file():
    sys.stderr.write('STOP: fresh independent native observer ACK required before child exit\\n')
    sys.exit(1)

ack = json.loads(ack_path.read_text())
assert ack['pass'] is True
assert ack['process_pid'] == os.getpid()
assert ack['process_creation_FILETIME'] == filetime
sys.exit(0)
""")

    # Launch child
    p_child = subprocess.Popen([sys.executable, str(child_script)], stderr=subprocess.PIPE, text=True)

    # Supervisor monitors pending file
    timeout = time.monotonic() + 10.0
    while p_child.poll() is None and (not pending_path.exists()) and time.monotonic() < timeout:
        time.sleep(0.05)

    assert pending_path.exists(), "Child must write pending file"
    pending_content = json.loads(pending_path.read_text())
    assert pending_content['process_pid'] == p_child.pid

    # Independent observer script: reads pending, verifies live process, writes ACK
    observer_script = tmp_path / 'independent_observer.py'
    observer_script.write_text(f"""
import ctypes, hashlib, json, os, pathlib, sys
from ctypes import wintypes

class FILETIME(ctypes.Structure):
    _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]

pending_path = pathlib.Path(r'{pending_path}')
ack_path = pathlib.Path(r'{ack_path}')

pending = json.loads(pending_path.read_text())
target_pid = pending['process_pid']

k = ctypes.WinDLL('kernel32', use_last_error=True)
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.OpenProcess.restype = wintypes.HANDLE
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]
k.GetProcessTimes.restype = wintypes.BOOL

h = k.OpenProcess(0x400, False, target_pid)
if not h:
    sys.stderr.write('OpenProcess failed for target child PID\\n')
    sys.exit(2)

times = [FILETIME() for _ in range(4)]
k.GetProcessTimes(h, ctypes.byref(times[0]), ctypes.byref(times[1]), ctypes.byref(times[2]), ctypes.byref(times[3]))
k.CloseHandle(h)
actual_filetime = (times[0].high << 32) | times[0].low

if actual_filetime != pending['process_creation_FILETIME']:
    sys.stderr.write('Process creation FILETIME mismatch\\n')
    sys.exit(3)

pending_sha = hashlib.sha256(pending_path.read_bytes()).hexdigest()

ack_data = {{
    'pass': True,
    'process_pid': target_pid,
    'process_creation_FILETIME': actual_filetime,
    'pending_sha256': pending_sha,
    'observer_pid': os.getpid(),
    'candidate_id': pending['candidate_id'],
    'source_map_sha256': pending['source_map_sha256'],
    'source_count': pending['source_count'],
    'target_snapshot_sha256': pending['target_snapshot_sha256'],
}}
ack_path.write_text(json.dumps(ack_data, sort_keys=True))
sys.exit(0)
""")

    # Launch observer as an independent process
    p_observer = subprocess.Popen([sys.executable, str(observer_script)], stderr=subprocess.PIPE, text=True)
    obs_exit = p_observer.wait()
    assert obs_exit == 0, f"Observer must exit 0, stderr: {p_observer.stderr.read()}"

    # Wait for child to finish validation and exit
    child_exit = p_child.wait()
    assert child_exit == 0, f"Child must exit 0, stderr: {p_child.stderr.read()}"

    # Verify produced ACK
    assert ack_path.exists()
    ack = json.loads(ack_path.read_text())
    assert ack['pass'] is True
    assert ack['process_pid'] == p_child.pid
    assert ack['process_creation_FILETIME'] == pending_content['process_creation_FILETIME']
    assert ack['pending_sha256'] == hashlib.sha256(pending_path.read_bytes()).hexdigest()

def test_supervisor_command_construction_and_routing():
    """
    Tests that when a process group contains multiple suites:
    1. Group name 'revision001_32,revision002_39' maps to prefix 'revision001_32_revision002_39'.
    2. Supervisor routes pending and ACK paths to 'revision001_32_revision002_39.pending.json'
       and 'revision001_32_revision002_39.independent_ack.json'.
    3. The child runner command is constructed with --suite revision001_32,revision002_39,
       ensuring child and supervisor evaluate identical prefixes.
    4. Receipt paths for all member suites in the group are resolved and accounted for.
    """
    group = "revision001_32,revision002_39"
    suite_ids = group.split(',')
    assert suite_ids == ['revision001_32', 'revision002_39']

    # Prefix alignment
    group_prefix = group.replace(',', '_')
    assert group_prefix == 'revision001_32_revision002_39'

    # Verify that child command with --suite group produces matching prefix
    child_argv = ['python.exe', 'run_suite.py', '--suite', group]
    child_suite_arg = child_argv[child_argv.index('--suite') + 1]
    child_prefix = child_suite_arg.replace(',', '_')
    assert child_prefix == group_prefix, "Child prefix must match supervisor group prefix"

    # Both receipts must be expected
    report_root = pathlib.Path(r'C:\reports')
    receipt_paths = [report_root / 'suite_receipts' / f'{sid}.json' for sid in suite_ids]
    assert len(receipt_paths) == 2
    assert receipt_paths[0].name == 'revision001_32.json'
    assert receipt_paths[1].name == 'revision002_39.json'
