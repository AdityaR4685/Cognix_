"""Local evidence checks and guards. No modeling or acquisition operations."""
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
TARGET = REPORT / 'partition_freeze_v1'
HEAD = '1e636800438a681ecfba7630ed372581abc4c8c7'
V8_SEAL = '257a2bf005b57f0bd8e1cd82789debedb5a72d27f0960b1a03e66cb6b1a1f810'
TOKEN = 'HUMAN_REVIEWED_EXPERIMENT_2B_PARTITION_FREEZE_V1'
SOURCE = {'path': r'E:\carlanomaly-base-train.tar.gz', 'bytes': 146453559283,
          'sha256': '6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683'}


class ReviewRequired(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReviewRequired(message + '; STOP FOR HUMAN REVIEW')


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def pretty_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Linked/junction path rejected: ' + str(path))
    for parent in (path, *path.parents):
        if parent.exists():
            require(not parent.is_symlink() and not
                    (getattr(parent.lstat(), 'st_file_attributes', 0) &
                     getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse path rejected')
    return path


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON constant'))


def verify_manifest(root, listing='SHA256SUMS', expected_seal=None, exact=True):
    root = safe_path(root)
    detached = (root / (listing + '.sha256')).read_bytes()
    raw = (root / listing).read_bytes()
    seal = sha_bytes(raw)
    require(detached == (seal + '  ' + listing + '\n').encode(), 'Detached seal mismatch')
    require(expected_seal is None or seal == expected_seal, 'Expected seal mismatch')
    names = set()
    for line in raw.decode('utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', digest) and Path(name).name == name and
                name not in names and name not in (listing, listing + '.sha256'), 'Invalid sealed path')
        names.add(name)
        path = safe_path(root / name)
        require(path.is_file() and hash_file(path) == digest, 'Sealed file mismatch: ' + name)
    require(bool(names), 'Empty manifest')
    if exact:
        require({p.name for p in root.iterdir()} == names | {listing, listing + '.sha256'},
                'Extra or missing bundle entry')
    return {'sha256': seal, 'files_verified': len(names)}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO).decode('utf-8').strip()


def fingerprint(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'device': s.st_dev, 'inode': s.st_ino,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def verify_source(identity, bound, progress=None):
    path = safe_path(identity['path'])
    require(path.is_absolute() and str(path) == identity['path'], 'Source path mismatch')
    before = fingerprint(path)
    require(before['bytes'] == identity['bytes'] and bound['identity'] == identity and
            bound['fingerprint'] == before and
            bound['independently_measured_full_sha256'] == identity['sha256'], 'Source identity mismatch')
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        key = lambda s: (s.st_size, s.st_dev, s.st_ino, s.st_mtime_ns, s.st_ctime_ns)
        require(key(opened)[:4] == tuple(before[k] for k in ('bytes', 'device', 'inode', 'mtime_ns')),
                'Opened source identity mismatch')
        for data in iter(lambda: stream.read(8 << 20), b''):
            count += len(data)
            h.update(data)
            if progress:
                progress(count)
        require(key(os.fstat(stream.fileno())) == key(opened), 'Source descriptor changed')
    require(fingerprint(path) == before and count == identity['bytes'] and
            h.hexdigest() == identity['sha256'], 'Source SHA256 mismatch or source changed')
    return dict(bound)


def evidence_files(roots):
    files, directories = [], []
    for relative in roots:
        root = safe_path(REPO / relative)
        for path in sorted([root, *root.rglob('*')]):
            # Roots/ancestors are checked once. Every descendant is checked with
            # lstat before reading, avoiding repeated ancestor syscalls per file.
            # A directory sorts before its descendants, so links fail before traversal reads.
            s = path.lstat()
            require(not path.is_symlink() and not (getattr(s, 'st_file_attributes', 0) &
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse evidence entry')
            record = {'path': path.relative_to(REPO).as_posix(), 'mtime_ns': s.st_mtime_ns}
            if stat.S_ISDIR(s.st_mode):
                directories.append(record)
            else:
                require(stat.S_ISREG(s.st_mode), 'Unexpected evidence file type')
                files.append(dict(record, bytes=s.st_size, sha256=hash_file(path)))
    return {'roots': roots, 'files': files, 'directories': directories}


def verify_preservation(baseline):
    actual = evidence_files(baseline['roots'])
    require(actual == baseline, 'Historical file/directory inventory, hash, size or timestamp changed')
    return {'files_verified': len(actual['files']), 'directories_verified': len(actual['directories']),
            'hashes_sizes_mtimes_and_exact_inventories_unchanged': True}


def make_guard(writable=(), allow_source=False, counters=None):
    counters = counters if counters is not None else {}
    for key in ('network_requests', 'TEST_requests', 'blocked_network_attempts',
                'blocked_TEST_attempts', 'blocked_mutation_attempts', 'blocked_source_attempts'):
        counters.setdefault(key, 0)
    norm = lambda p: os.path.normcase(os.path.abspath(os.fsdecode(p)))
    roots = [norm(p) for p in writable]
    source = norm(SOURCE['path'])
    def audit(event, args):
        if event.startswith('socket.') and event != 'socket.gethostname':
            counters['blocked_network_attempts'] += 1
            require(False, 'Network operation forbidden')
        if event == 'subprocess.Popen':
            command = args[1]
            if isinstance(command, str):
                command = [s.strip('"') for s in shlex.split(command, posix=False)]
            ok = isinstance(command, (list, tuple)) and len(command) > 1 and \
                Path(str(command[0])).name.lower() in ('git', 'git.exe') and \
                command[1] in ('show', 'diff', 'status', 'rev-parse', 'ls-files')
            require(ok, 'Only read-only local Git subprocess allowed')
        def check(name, write=False):
            if not isinstance(name, (str, bytes, os.PathLike)):
                return
            name = norm(name)
            parts = Path(name).parts
            if 'test' in parts or 'carlanomaly-base-test' in name or Path(name).name == 'anomaly-observation.feather':
                counters['blocked_TEST_attempts'] += 1
                require(False, 'TEST/label access forbidden')
            if name == source and (write or not allow_source):
                counters['blocked_source_attempts'] += 1
                require(False, 'Source access forbidden in this phase')
            if write and not any(name == r or name.startswith(r + os.sep) for r in roots):
                counters['blocked_mutation_attempts'] += 1
                require(False, 'Mutation outside authorized new output forbidden')
        if event == 'open':
            mode, flags = args[1], args[2]
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or \
                bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            check(args[0], write)
        if event in ('os.remove', 'os.rmdir', 'os.rename', 'os.mkdir', 'os.chmod',
                     'os.utime', 'os.link', 'os.symlink', 'shutil.copyfile'):
            count = 2 if event in ('os.rename', 'os.link', 'os.symlink') else 1
            for value in args[:count]:
                check(value, True)
    return audit, counters


def install_guard(writable=(), allow_source=False):
    guard, counters = make_guard(writable, allow_source)
    sys.addaudithook(guard)
    return counters


def write_new(path, raw):
    path = safe_path(path)
    with path.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'Short output write')
        stream.flush()
        os.fsync(stream.fileno())
