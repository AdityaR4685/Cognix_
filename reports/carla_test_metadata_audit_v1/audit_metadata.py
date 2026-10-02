"""Metadata audit tooling. Standard library only; never import COGNIX/models."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRE = ROOT / 'reports/carla_final_evaluation_preregistration_v1'
EXPECTED = 'e03da4019c35d53420e42e02d4c0ebe2dd539a24ff5b645fdf5427078122ade1'

def now():
    return datetime.now(timezone.utc).isoformat()

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()

def seal_check(directory, name='SHA256SUMS'):
    manifest = directory / name
    detached = manifest.with_name(name + '.sha256')
    if detached.exists():
        assert sha(manifest) == detached.read_text().split()[0], str(detached)
    records = []
    for line in manifest.read_text().splitlines():
        expected, rel = line.split('  ', 1)
        target = (directory / rel).resolve()
        assert target.is_relative_to(directory.resolve())
        actual = sha(target)
        records.append({'path': str(target), 'expected': expected, 'actual': actual})
        assert actual == expected, str(target)
    return {'manifest': str(manifest), 'sha256': sha(manifest), 'files_verified': len(records), 'passed': True}

def preflight():
    ledger = {'started_utc': now(), 'entries': [], 'official_metadata_accesses': [],
              'scope': 'Preflight integrity reads only. Opaque hashes of already completed development artifacts are not interpreted as predictions or metrics. No official TEST files, archive bytes, sensor members or labels may be read.',
              'attestation_limit': 'Executed-action ledger, not an OS-wide trace.'}
    write('metadata_access_ledger.json', ledger)
    initial = read(PRE / 'amendment_initial_state.json')
    snapshot = read(PRE / 'preservation_snapshot.json')
    head = git('rev-parse', 'HEAD')
    parent = git('rev-parse', 'HEAD^')
    changes = git('diff', '--name-only', initial['git']['HEAD'], 'HEAD').splitlines()
    assert parent == initial['git']['HEAD']
    assert changes and all(p.startswith('reports/carla_final_evaluation_preregistration_v1/') for p in changes)
    assert git('diff', '--name-only', 'HEAD') == '', 'Tracked working tree changes'
    assert sha(PRE / 'SHA256SUMS') == EXPECTED
    completed = subprocess.check_output([sys.executable, '-B', str(PRE / 'verify_protocol.py'), 'verify'], cwd=ROOT)
    result = {'started_utc': now(), 'current_git_commit': head, 'recorded_pre_sealing_commit': parent,
              'commit_reconciliation': 'Current HEAD is the direct child committing only the sealed preregistration. Scientific tree is unchanged from the recorded pre-sealing HEAD. No arbitrary HEAD rebinding.',
              'commit_changes': changes, 'protocol_verification': json.loads(completed), 'seals': [], 'failures': []}
    for rec in snapshot['seals']:
        p = Path(rec['path'])
        checked = seal_check(p.parent, p.name)
        assert checked['sha256'] == rec['sha256']
        result['seals'].append(checked)
    print('All completed-run, graph-protocol and environment seals verified', flush=True)
    hashes = {}
    for name, expected in initial['workspace_sha256'].items():
        actual = sha(ROOT / name)
        hashes[name] = actual
        if actual != expected:
            result['failures'].append({'path': name, 'expected': expected, 'actual': actual})
    external = {}
    for name, rec in initial['external_artifact_hashes'].items():
        actual = sha(name)
        external[name] = actual
        if actual != rec['expected']:
            result['failures'].append({'path': name, 'expected': rec['expected'], 'actual': actual})
    result['workspace_files_verified'] = len(hashes)
    result['external_files_verified'] = len(external)
    result['passed'] = not result['failures']
    result['completed_utc'] = now()
    write('preregistration_verification.json', result)
    assert result['passed'], 'Preflight mismatch; STOP before metadata'
    pre_hashes = {p.relative_to(PRE).as_posix(): sha(p) for p in PRE.rglob('*') if p.is_file()}
    write('preservation_snapshot.json', {'git_HEAD': head, 'git_status': git('status', '--porcelain=v1', '--untracked-files=all'),
          'workspace_hashes': hashes, 'external_hashes': external, 'preregistration_hashes': pre_hashes,
          'TRAIN_integrity_baseline': 'Sealed current_train_content_verification.json; raw TRAIN content will not be reopened for metadata audit.', 'time_utc': now()})
    ledger['entries'].append({'order': 1, 'timestamp_utc': now(), 'source': 'sealed local preregistration and frozen development bindings',
        'operation': 'SHA256 verification and Git tree/commit verification', 'path': str(PRE),
        'fields_inspected': ['seal entries', 'exact exclusions', 'partition algorithm', 'conformal rank rule', 'binding hashes', 'Git parent/tree'],
        'why_metadata': 'Preflight and immutable-artifact integrity; no official TEST or sensor/label payload',
        'manifest_sha256': EXPECTED, 'workspace_files_hashed': len(hashes), 'external_files_hashed': len(external)})
    write('metadata_access_ledger.json', ledger)
    print(json.dumps({k: result[k] for k in ['passed', 'current_git_commit', 'workspace_files_verified', 'external_files_verified']}))

if __name__ == '__main__':
    if sys.argv[1] == 'preflight':
        preflight()
