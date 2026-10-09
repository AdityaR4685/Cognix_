"""Gate-2 local-only bindings, exact seals and fail-closed process guards."""
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
REPO = BUNDLE.parents[2]
REPORT = BUNDLE.parent
V8 = REPORT / 'gate1_execution_bundle_v8'
STORE = REPORT / 'gate1_local_extraction_v8'
PARTITION_BUNDLE = REPORT / 'partition_freeze_bundle_v1'
PARTITION = REPORT / 'partition_freeze_v1'
TARGET = REPORT / 'gate2_train_health_v3'
V2 = REPORT / 'gate2_execution_bundle_v2'
V2_RUNTIME = REPORT / 'gate2_train_health_v2'
V2_SEAL = 'd268be110ba02d86523e35f002bbcd4164b139811244bbdd2cde54979d9e212c'
V1 = REPORT / 'gate2_execution_bundle_v1'
V1_RUNTIME = REPORT / 'gate2_train_health_v1'
V1_SEAL = '9d1ddc92177718cae655184cf4b5409078034c1fdbde379be90f30d802a24ec2'
V7 = REPORT / 'gate1_execution_bundle_v7'
V7_STORE = REPORT / 'gate1_local_extraction_v7'
V7_SEAL = '3750f41b8203550ba1f9c5b3d70330407ce7d2db5f99b4287d0f6cac9b66e4d8'
HEAD = '1e636800438a681ecfba7630ed372581abc4c8c7'
V8_SEAL = '257a2bf005b57f0bd8e1cd82789debedb5a72d27f0960b1a03e66cb6b1a1f810'
PARTITION_SHA = '706c819ec568fd2dd2a1a89d9d7cc13eed7ba86466d8b5aa4ad4de277a863230'
PARTITION_SEAL = 'bd257b2d2d5d4c9b28ea12419418bdea0f9b03473757360e538abca61ce0550d'
PARTITION_BUNDLE_SEAL = 'e925e480a103c26e21bd5d3bca8eabf0f61db0485dd852732208f5320a05935d'
TOKEN = 'HUMAN_REVIEWED_EXPERIMENT_2B_GATE2_TRAIN_HEALTH_V3'
SOURCE = {'path': r'E:\carlanomaly-base-train.tar.gz', 'bytes': 146453559283,
          'sha256': '6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683'}
MODALITIES = ('Camera', 'Seg', 'IMU')
DIMS = {'Camera': 18, 'Seg': 29, 'IMU': 10}
RECIPES = ('camera_brightness_shift', 'camera_occlusion', 'seg_region_corruption',
           'gnss_bias', 'gnss_drift', 'imu_spike', 'imu_bias_scale')
SEVERITIES = {'camera_brightness_shift': .35, 'camera_occlusion': .25,
              'seg_region_corruption': .2, 'imu_spike': 15., 'imu_bias_scale': .5}
SEMANTICS = 'P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo calibration task'


class ReviewRequired(RuntimeError):
    pass


def require(predicate, message):
    if not predicate:
        raise ReviewRequired(message + '; STOP FOR HUMAN REVIEW')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def pretty(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(8 << 20), b''):
            h.update(raw)
    return h.hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Linked/junction path rejected')
    for item in (path, *path.parents):
        if item.exists():
            require(not item.is_symlink() and not (getattr(item.lstat(), 'st_file_attributes', 0) &
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse path rejected')
    return path


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON constant'))


def write_new(path, raw):
    with safe_path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw), 'Short write')
        stream.flush()
        os.fsync(stream.fileno())


def listing_bytes(root, excluded=('SHA256SUMS', 'SHA256SUMS.sha256')):
    root = safe_path(root)
    records = []
    for path in sorted(root.rglob('*')):
        safe_path(path)
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            records.append(hash_file(path) + '  ' + path.relative_to(root).as_posix() + '\n')
    require(records, 'Empty seal')
    return ''.join(records).encode()


def seal_tree(root):
    # Interrupted work can include empty raw/pending directories. Bind their
    # exact inventory rather than delete or silently ignore forensic evidence.
    directories = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir())
    directory_record = root / 'DIRECTORY_INVENTORY.json'
    if directories or directory_record.exists():
        raw_directories = canonical(directories)
        if directory_record.exists():
            require(directory_record.read_bytes() == raw_directories, 'Directory inventory mismatch')
        else:
            write_new(directory_record, raw_directories)
    raw = listing_bytes(root)
    listing, detached = root / 'SHA256SUMS', root / 'SHA256SUMS.sha256'
    if listing.exists():
        require(listing.read_bytes() == raw, 'Partial publication listing mismatch')
    else:
        write_new(listing, raw)
    seal = digest(raw)
    record = (seal + '  SHA256SUMS\n').encode()
    if detached.exists():
        require(detached.read_bytes() == record, 'Partial publication detached mismatch')
    else:
        write_new(detached, record)
    verify_seal(root, seal)
    return seal


def verify_seal(root, expected=None):
    root = safe_path(root)
    raw = (root / 'SHA256SUMS').read_bytes()
    actual = digest(raw)
    require(expected is None or actual == expected, 'Expected seal mismatch')
    require((root / 'SHA256SUMS.sha256').read_bytes() == (actual + '  SHA256SUMS\n').encode(),
            'Detached seal mismatch')
    paths = set()
    for line in raw.decode().splitlines():
        sha, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', sha) and '\\' not in name and
                all(p not in ('', '.', '..') for p in name.split('/')) and
                not Path(name).is_absolute() and ':' not in name and name not in paths and
                name not in ('SHA256SUMS', 'SHA256SUMS.sha256'), 'Invalid sealed path')
        paths.add(name)
        path = safe_path(root / name)
        require(path.is_file() and hash_file(path) == sha, 'Sealed file mismatch: ' + name)
    require(raw == listing_bytes(root), 'Unexpected/missing sealed file')
    # No empty or extra directories can hide unsealed state.
    expected_dirs = {str(Path(n).parent).replace('\\', '/') for n in paths}
    expected_dirs |= {str(p).replace('\\', '/') for n in paths for p in Path(n).parents}
    actual_dirs = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_dir())
    if (root / 'DIRECTORY_INVENTORY.json').exists():
        require(read_json(root / 'DIRECTORY_INVENTORY.json') == actual_dirs, 'Sealed directory inventory mismatch')
    else:
        require(all(name in expected_dirs for name in actual_dirs), 'Unexpected empty directory')
    return {'seal': actual, 'files_verified': len(paths)}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO).decode().strip()


def make_guard(writable=(), allow_source=False):
    counters = dict(network_requests=0, TEST_requests=0, blocked_network_attempts=0,
                    blocked_TEST_attempts=0, blocked_graph_attempts=0,
                    blocked_mutation_attempts=0, blocked_source_attempts=0)
    norm = lambda p: os.path.normcase(os.path.abspath(os.fsdecode(p)))
    roots = [norm(p) for p in writable]
    source = norm(SOURCE['path'])
    def check(path, write=False):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        name = norm(path)
        if name == norm(os.devnull):
            return  # Windows NUL is a device; platform's read-only version probe uses it.
        parts = Path(name).parts
        if any(p in ('test', 'test_normal', 'test_anomaly') for p in parts) or \
                'carlanomaly-base-test' in name or Path(name).name == 'anomaly-observation.feather':
            counters['blocked_TEST_attempts'] += 1
            require(False, 'TEST/official label access forbidden')
        if name == source and (write or not allow_source):
            counters['blocked_source_attempts'] += 1
            require(False, 'Real source forbidden in synthetic phase')
        if write and not any(name == root or name.startswith(root + os.sep) for root in roots):
            counters['blocked_mutation_attempts'] += 1
            require(False, 'Mutation outside new authorized output forbidden')
    def audit(event, args):
        if event.startswith('socket.') and event != 'socket.gethostname':
            counters['blocked_network_attempts'] += 1
            require(False, 'Network forbidden')
        if event == 'import' and (str(args[0]).startswith(('cognix.graph', 'torch')) or
                                 'graph_fit_export' in str(args[0]) or 'graph_train' in str(args[0])):
            counters['blocked_graph_attempts'] += 1
            require(False, 'Graph/GAT import forbidden')
        if event == 'subprocess.Popen':
            command = args[1]
            if isinstance(command, str):
                command = [s.strip('"') for s in shlex.split(command, posix=False)]
            version_probe = isinstance(command, (list, tuple)) and len(command) == 3 and \
                norm(command[0]) == norm(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'cmd.exe') and \
                str(command[1]).lower() == '/c' and str(command[2]).strip('"').lower() == 'ver'
            read_only_git = (isinstance(command, (list, tuple)) and len(command) > 1 and
                    Path(str(command[0])).name.lower() in ('git', 'git.exe') and
                    command[1] in ('show', 'diff', 'status', 'rev-parse', 'ls-files'))
            require(read_only_git or version_probe, 'Only read-only Git or exact Windows version probe permitted')
        if event == 'open':
            mode, flags = args[1:3]
            check(args[0], (isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                  bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.chmod', 'os.utime',
                     'os.link', 'os.symlink', 'shutil.copyfile'):
            for path in args[:2 if event in ('os.rename', 'os.link', 'os.symlink', 'shutil.copyfile') else 1]:
                check(path, True)
    return audit, counters


def install_guard(writable=(), allow_source=False):
    audit, counters = make_guard(writable, allow_source)
    sys.addaudithook(audit)
    return counters


def authenticate_upstream():
    verify_seal(PARTITION_BUNDLE, PARTITION_BUNDLE_SEAL)
    verify_seal(V8, V8_SEAL)
    from gate2_resolution import authenticate_verifiers
    authenticate_verifiers()
    for path in PARTITION_BUNDLE.glob('*.py'):
        module = sys.modules.get(path.stem)
        require(module is None or Path(getattr(module, '__file__', '')).resolve() == path,
                'Ambiguous partition verifier import: ' + path.stem)
    sys.path.append(str(PARTITION_BUNDLE))
    import importlib.util
    for path in PARTITION_BUNDLE.glob('*.py'):
        spec = importlib.util.find_spec(path.stem)
        require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == path,
                'Ambiguous partition verifier discovery: ' + path.stem)
    import partition_evidence
    require(Path(partition_evidence.__file__).resolve().parent == PARTITION_BUNDLE,
            'Wrong upstream verifier location')
    return partition_evidence
