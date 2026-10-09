"""Independent offline evidence primitives. Never import Gate-1/2 execution code."""
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
REPORT = BUNDLE.parent
REPO = REPORT.parents[1]
TARGET = REPORT / 'graph_partition_freeze_v1'
PENDING = REPORT / '.graph_partition_freeze_v1.pending'
HEAD = '1e636800438a681ecfba7630ed372581abc4c8c7'
UPSTREAM_SEALS = {
    'gate1_execution_bundle_v8': '257a2bf005b57f0bd8e1cd82789debedb5a72d27f0960b1a03e66cb6b1a1f810',
    'partition_freeze_bundle_v1': 'e925e480a103c26e21bd5d3bca8eabf0f61db0485dd852732208f5320a05935d',
    'partition_freeze_v1': 'bd257b2d2d5d4c9b28ea12419418bdea0f9b03473757360e538abca61ce0550d',
    'gate2_execution_bundle_v3': '2e686caf9a6dbe6db99c1911130b5b94cf0fea14e73be8cf59b92fc2f8031b7f',
    'gate2_train_health_v3': '58cc01e1203b697e201276e044e69b118063801d9300569f59df97e74f0f912a',
}
PARTITION_SHA = '706c819ec568fd2dd2a1a89d9d7cc13eed7ba86466d8b5aa4ad4de277a863230'
FINAL_SHA = '8cbb80b10464bd38eb6ef567857d071c11313a8bda937c1ded4e27f20bfca320'
SOURCE = {'path': r'E:\carlanomaly-base-train.tar.gz', 'bytes': 146453559283,
          'sha256': '6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683'}
ZERO_STATE = {'TEST_requests': 0, 'network_requests': 0,
              'graph_constructed': False, 'GAT_executed': False}
LATER_SPEC = {
    'node_order': ['Camera', 'IMU', 'Seg'],
    'node_feature_order': ['prob_normal', 'epistemic', 'aleatoric'],
    'graph_conditions': ['NoGraph', 'StandardGAT', 'EpistemicGAT'],
    'epistemic_prior': {'w_j': '1 / (1 + E_j)', 'e_ij^epi': 'e_ij - log(1 + E_j)'},
    'execution_authorized': False,
}


class FreezeError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise FreezeError(message + '; STOP FOR HUMAN REVIEW')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON'))


def safe_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Linked path rejected: ' + str(path))
    for part in (path, *path.parents):
        if part.exists():
            require(not part.is_symlink() and not (getattr(part.lstat(), 'st_file_attributes', 0) &
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse path rejected')
    return path


def hash_file(path):
    h = hashlib.sha256()
    with safe_path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(8 << 20), b''):
            h.update(raw)
    return h.hexdigest()


def write_new(path, raw):
    with safe_path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw), 'Short write')
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, obj):
    write_new(path, canonical(obj))


def tree_inventory(root, exclude=()):
    root = safe_path(root)
    files, dirs = [], []
    for path in sorted(root.rglob('*')):
        safe_path(path)
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            dirs.append(name)
        elif path.is_file() and name not in exclude:
            files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': hash_file(path)})
        else:
            require(path.is_file(), 'Nonregular inventory entry')
    return {'files': files, 'directories': dirs}


def seal_tree(root):
    # Inventory excludes itself and seal metadata to avoid a hash cycle. SHA256SUMS
    # covers the inventory itself and every payload, with an exact file set check.
    inv = tree_inventory(root, ('artifact_inventory.json', 'SHA256SUMS', 'SHA256SUMS.sha256'))
    write_json(root / 'artifact_inventory.json', inv)
    full = tree_inventory(root, ('SHA256SUMS', 'SHA256SUMS.sha256'))
    raw = ''.join(f"{r['sha256']}  {r['path']}\n" for r in full['files']).encode()
    write_new(root / 'SHA256SUMS', raw)
    seal = digest(raw)
    write_new(root / 'SHA256SUMS.sha256', (seal + '  SHA256SUMS\n').encode())
    verify_seal(root, seal)
    return seal


def verify_seal(root, expected):
    root = safe_path(root)
    require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Invalid expected seal')
    raw = (root / 'SHA256SUMS').read_bytes()
    require(digest(raw) == expected, 'Expected seal mismatch: ' + root.name)
    require((root / 'SHA256SUMS.sha256').read_bytes() == (expected + '  SHA256SUMS\n').encode(),
            'Detached seal mismatch: ' + root.name)
    names = set()
    for line in raw.decode('utf-8').splitlines():
        sha, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', sha) is not None and '\\' not in name and
                ':' not in name and not name.startswith('/') and
                all(p not in ('', '.', '..') for p in name.split('/')) and name not in names and
                name not in ('SHA256SUMS', 'SHA256SUMS.sha256'), 'Invalid manifest entry')
        names.add(name)
        require(hash_file(root / name) == sha, 'Sealed file mismatch: ' + str(root / name))
    actual = tree_inventory(root, ('SHA256SUMS', 'SHA256SUMS.sha256'))
    generated = ''.join(f"{r['sha256']}  {r['path']}\n" for r in actual['files']).encode()
    require(raw == generated, 'Extra/missing file or noncanonical manifest: ' + root.name)
    if (root / 'DIRECTORY_INVENTORY.json').exists():
        require(read_json(root / 'DIRECTORY_INVENTORY.json') == sorted(actual['directories']),
                'Upstream directory inventory mismatch')
    else:
        expected_dirs = {p.as_posix() for n in names for p in Path(n).parents if p != Path('.')}
        require(set(actual['directories']) == expected_dirs, 'Unexpected empty/missing directory')
    if (root / 'artifact_inventory.json').exists():
        require(read_json(root / 'artifact_inventory.json') == tree_inventory(root,
                ('artifact_inventory.json', 'SHA256SUMS', 'SHA256SUMS.sha256')), 'Inventory mismatch')
    return {'path': root.relative_to(REPO).as_posix() if root.is_relative_to(REPO) else str(root),
            'seal_sha256': expected, 'files_verified': len(names), 'all_payloads_hashed': True}


def git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    return subprocess.check_output(['git', *args], cwd=REPO, env=env).decode().strip()


def git_state():
    return {'HEAD': git('rev-parse', 'HEAD'),
            'status_porcelain': git('status', '--porcelain=v1', '--untracked-files=normal'),
            'tracked_changes': git('diff', '--name-only'),
            'staged_changes': git('diff', '--cached', '--name-only')}


def verify_git_preserved(current, initial):
    require(all(current[k] == initial[k] for k in ('HEAD', 'tracked_changes', 'staged_changes')),
            'Git HEAD/tracked/staged state changed')
    allowed = {'?? ' + p.relative_to(REPO).as_posix() + '/' for p in (TARGET, PENDING)}
    def prior_status(state):
        return [line for line in state['status_porcelain'].splitlines() if line not in allowed]
    require(prior_status(current) == prior_status(initial), 'Prior Git untracked state changed')


def source_fingerprint():
    s = safe_path(SOURCE['path']).stat()
    return {'bytes': s.st_size, 'device': s.st_dev, 'inode': s.st_ino,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def assert_absent(target=TARGET, pending=PENDING):
    require(not safe_path(target).exists(), 'Immutable target already exists')
    require(not safe_path(pending).exists() and
            not list(target.parent.glob('.' + target.name + '.pending*')), 'Pending evidence exists')


def make_guard(writable=()):
    """Offline audit guard, including writes, subprocesses and raw data access."""
    counters = dict(ZERO_STATE, blocked_TEST_attempts=0, blocked_network_attempts=0,
                    blocked_mutation_attempts=0, blocked_model_attempts=0,
                    blocked_raw_archive_attempts=0, blocked_subprocess_attempts=0)
    norm = lambda p: os.path.normcase(os.path.abspath(os.fsdecode(p)))
    roots = [norm(p) for p in writable]
    source = norm(SOURCE['path'])

    def check_path(path, write=False):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        name = norm(path)
        if name == norm(os.devnull):
            return
        parts = Path(name).parts
        if any(p in ('test', 'test_normal', 'test_anomaly') for p in parts) or \
                'carlanomaly-base-test' in name or Path(name).name == 'anomaly-observation.feather':
            counters['blocked_TEST_attempts'] += 1
            require(False, 'TEST/official-label access forbidden')
        if name == source:
            counters['blocked_raw_archive_attempts'] += 1
            require(False, 'Raw archive access unnecessary and forbidden; stat binding only')
        if write and not any(name == root or name.startswith(root + os.sep) for root in roots):
            counters['blocked_mutation_attempts'] += 1
            require(False, 'Write outside new output namespace forbidden')

    def guard(event, args):
        if event.startswith('socket.'):
            counters['blocked_network_attempts'] += 1
            require(False, 'Network forbidden')
        if event == 'import' and str(args[0]).split('.')[0] in ('torch', 'tensorflow', 'jax', 'cognix'):
            counters['blocked_model_attempts'] += 1
            require(False, 'Model/graph implementation import forbidden')
        if event == 'subprocess.Popen':
            command = args[1]
            if isinstance(command, str):
                command = [p.strip('"') for p in shlex.split(command, posix=False)]
            is_git = isinstance(command, (list, tuple)) and len(command) > 1 and \
                Path(command[0]).name.lower() in ('git', 'git.exe') and tuple(command[1:]) in (
                    ('rev-parse', 'HEAD'), ('status', '--porcelain=v1', '--untracked-files=normal'),
                    ('diff', '--name-only'), ('diff', '--cached', '--name-only'))
            is_replay = command == [sys.executable, '-B', str(BUNDLE / 'replay_partition.py')]
            if not (is_git or is_replay):
                counters['blocked_subprocess_attempts'] += 1
                require(False, 'Unapproved subprocess forbidden')
        if event == 'open':
            mode, flags = args[1:3]
            check_path(args[0], (isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                       bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.chmod', 'os.utime',
                     'os.link', 'os.symlink', 'shutil.copyfile'):
            for path in args[:2 if event in ('os.rename', 'os.link', 'os.symlink', 'shutil.copyfile') else 1]:
                check_path(path, True)
    return guard, counters


def install_guard(writable=()):
    guard, counters = make_guard(writable)
    sys.addaudithook(guard)
    return counters


def preservation_snapshot():
    """Every pre-existing file/directory in the full TRAIN reports namespace."""
    records, directories = [], []
    safe_path(REPORT)
    for top in sorted(REPORT.iterdir()):
        if top.name in (BUNDLE.name, TARGET.name, PENDING.name):
            continue
        safe_path(top)
        # Parents precede descendants in this sorted traversal. Admit every
        # directory and leaf individually; do not repeatedly resolve all already
        # admitted ancestors for thousands of files. Reject every reparse entry.
        for path in [top, *sorted(top.rglob('*'))] if top.is_dir() else [top]:
            rel = path.relative_to(REPO).as_posix()
            s = path.lstat()
            require(not (getattr(s, 'st_file_attributes', 0) &
                         getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse preservation entry')
            if stat.S_ISDIR(s.st_mode):
                directories.append(rel)
            else:
                require(stat.S_ISREG(s.st_mode), 'Nonregular preservation entry')
                h = hashlib.sha256()
                identity = lambda v: (v.st_size, v.st_dev, v.st_ino, v.st_mtime_ns)
                with path.open('rb') as stream:
                    require(identity(os.fstat(stream.fileno())) == identity(s), 'Preserved file open identity mismatch')
                    for raw in iter(lambda: stream.read(8 << 20), b''):
                        h.update(raw)
                    require(identity(os.fstat(stream.fileno())) == identity(s), 'Preserved file changed during hash')
                require(identity(path.lstat()) == identity(s), 'Preserved path changed during hash')
                records.append({'path': rel, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
                                'ctime_ns': s.st_ctime_ns, 'device': s.st_dev, 'inode': s.st_ino,
                                'sha256': h.hexdigest()})
    return {'schema': 'exact-historical-preservation-v1', 'root': REPORT.relative_to(REPO).as_posix(),
            'files': records, 'directories': directories, 'archive_fingerprint': source_fingerprint(),
            'scope': 'All pre-existing full-TRAIN report files, earlier bundles/runtimes/failure evidence; '
                     'archive identity metadata; Git HEAD/tracked/staged state separately verified',
            'access_times_excluded': 'Read access may update access time; no content/write metadata excluded'}


def verify_preservation(baseline):
    now = preservation_snapshot()
    require(now == baseline, 'Historical evidence or archive identity changed')
    return {'status': 'PASS', 'files_verified': len(now['files']),
            'directories_verified': len(now['directories']), 'all_file_sha256_equal': True,
            'all_sizes_and_write_metadata_equal': True, 'archive_fingerprint_equal': True,
            'added_historical_files': [], 'removed_historical_files': [], 'changed_historical_files': [],
            'baseline_sha256': digest(canonical(baseline))}


def environment():
    import numpy as np
    return {'python': sys.version, 'python_version': '.'.join(map(str, sys.version_info[:3])),
            'python_executable': sys.executable, 'python_executable_sha256': hash_file(sys.executable),
            'numpy_version': np.__version__, 'numpy_location': str(Path(np.__file__).resolve()),
            'platform': sys.platform}
