"""Offline Experiment-2B boundaries, independent SHA256 seals and exact preservation."""
import contextvars
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
REPORT = BUNDLE.parent
REPO = REPORT.parents[1]
DATA = REPORT / 'graph_development_data_v3'
RUNS = REPORT / 'graph_development_runs_v3'
PENDING = REPORT / '.graph_development_data_v3.pending'
V1_BUNDLE = REPORT / 'graph_development_bundle_v1'
V1_BUNDLE_SEAL = '7da8ffb4ded5ee446f07b369577a9f62f5191592c9da682f680ac76705fb8ece'
V1_PENDING = REPORT / '.graph_development_data_v1.pending'
V2_BUNDLE = REPORT / 'graph_development_bundle_v2'
V2_BUNDLE_SEAL = 'b0bb3ff9bdb33732442de9b8ea47b415039307612e13924f39c520bc02406c6a'
HEAD = '1e636800438a681ecfba7630ed372581abc4c8c7'
SOURCE = {'path': r'E:\carlanomaly-base-train.tar.gz', 'bytes': 146453559283,
          'sha256': '6cf22ecf7d2910b45ed71127525834d1a4a26a65cc65918e7b5e27da95a38683'}
SEALS = {
 'gate1_execution_bundle_v7': '3750f41b8203550ba1f9c5b3d70330407ce7d2db5f99b4287d0f6cac9b66e4d8',
 'gate1_execution_bundle_v8': '257a2bf005b57f0bd8e1cd82789debedb5a72d27f0960b1a03e66cb6b1a1f810',
 'partition_freeze_bundle_v1': 'e925e480a103c26e21bd5d3bca8eabf0f61db0485dd852732208f5320a05935d',
 'partition_freeze_v1': 'bd257b2d2d5d4c9b28ea12419418bdea0f9b03473757360e538abca61ce0550d',
 'gate2_execution_bundle_v3': '2e686caf9a6dbe6db99c1911130b5b94cf0fea14e73be8cf59b92fc2f8031b7f',
 'gate2_train_health_v3': '58cc01e1203b697e201276e044e69b118063801d9300569f59df97e74f0f912a',
 'graph_partition_freeze_bundle_v1': '01069701997349697ae303ce43327c84fdf22f09ce46156a1dfb09db917d68c1',
 'graph_partition_freeze_v1': '96ca1ceb098d363d20165d8ae6d0a2f7ab52c25dfac3e54b71b9d67ef5e9507f',
}
ORIGINAL_PARTITION_SHA = '706c819ec568fd2dd2a1a89d9d7cc13eed7ba86466d8b5aa4ad4de277a863230'
GRAPH_PARTITION_SHA = '78cad27a586a7e1cb675b242c7f57b7a11465da3916f7e0afc9e04df7da58724'
FINAL_SHA = '8cbb80b10464bd38eb6ef567857d071c11313a8bda937c1ded4e27f20bfca320'
PROTOCOL = REPO / 'reports/carla_gat_preregistration_v1/protocol.json'
PROTOCOL_SHA = '68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203'
HISTORICAL_EXPORTER = REPO / 'cognix/adapters/carla/graph_fit_export.py'
DATA_TOKEN = 'HUMAN_REVIEWED_EXPERIMENT_2B_FULL_TRAIN_GRAPH_DATA_V3'
NODE_ORDER = ('Camera', 'IMU', 'Seg')
FEATURE_ORDER = ('prob_normal', 'epistemic', 'aleatoric')
DIMS = {'Camera': 18, 'IMU': 10, 'Seg': 29}
METHODS = ('NoGraph', 'StandardGAT', 'EpistemicGAT')
SEEDS = (101, 202, 303, 404, 505)
RECIPES = ('camera_brightness_shift', 'camera_occlusion', 'seg_region_corruption', 'imu_spike', 'imu_bias_scale')
SEVERITIES = dict(zip(RECIPES, (.35, .25, .2, 15., .5)))
ADJACENCY_LIST = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
ZERO = dict(real_pseudo_generated=False, graph_artifact_constructed=False, NoGraph_trained=False,
            StandardGAT_trained=False, EpistemicGAT_trained=False, TEST_requests=0, network_requests=0)
_INTEGRITY = contextvars.ContextVar('graph_integrity_read', default=None)
ACTIVE_PHASE = None


class PreparationError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise PreparationError(message + '; STOP FOR HUMAN REVIEW')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path, 'Path alias/link rejected')
    for item in (path, *path.parents):
        if item.exists():
            require(not item.is_symlink() and not (getattr(item.lstat(), 'st_file_attributes', 0) &
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)), 'Reparse path rejected')
    return path


def norm(path):
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


def hash_file(path, *, source_progress=False, admitted=False):
    path = Path(path).absolute() if admitted else safe_path(path)
    before = path.stat()
    identity = lambda s: (s.st_size, s.st_dev, s.st_ino, s.st_mtime_ns)
    token = _INTEGRITY.set(norm(path))
    try:
        h, count, last = hashlib.sha256(), 0, time.monotonic()
        with path.open('rb') as stream:
            require(identity(os.fstat(stream.fileno())) == identity(before), 'Opened file identity changed')
            for raw in iter(lambda: stream.read(8 << 20), b''):
                h.update(raw); count += len(raw)
                if source_progress and time.monotonic() - last >= 25:
                    print('Read-only TRAIN archive integrity hash: %d / %d bytes' % (count, before.st_size), flush=True)
                    last = time.monotonic()
            require(identity(os.fstat(stream.fileno())) == identity(before), 'File changed during integrity hash')
        require(identity(path.stat()) == identity(before) and count == before.st_size, 'File changed after integrity hash')
        return h.hexdigest()
    finally:
        _INTEGRITY.reset(token)


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(safe_path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON constant'))


def write_new(path, raw):
    with safe_path(path).open('xb') as stream:
        require(stream.write(raw) == len(raw), 'Short write')
        stream.flush(); os.fsync(stream.fileno())


def write_json(path, value):
    write_new(path, canonical(value))


def inventory(root, exclude=()):
    safe_path(root)
    files, dirs = [], []
    for path in sorted(root.rglob('*'), key=lambda p:p.relative_to(root).as_posix()):
        safe_path(path)
        name = path.relative_to(root).as_posix()
        if path.is_dir(): dirs.append(name)
        elif path.is_file():
            if name not in exclude:
                files.append(dict(path=name, bytes=path.stat().st_size, sha256=hash_file(path)))
        else: require(False, 'Nonregular inventory entry')
    return dict(files=files, directories=dirs)


def verify_seal(root, expected):
    safe_path(root)
    require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'Malformed external seal')
    raw = (root / 'SHA256SUMS').read_bytes()
    require(digest(raw) == expected and (root / 'SHA256SUMS.sha256').read_bytes() ==
            (expected + '  SHA256SUMS\n').encode(), 'Seal/detached seal mismatch: ' + root.name)
    seen = set()
    for line in raw.decode().splitlines():
        sha, name = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', sha) is not None and '\\' not in name and ':' not in name and
                not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')) and
                name not in seen and name not in ('SHA256SUMS', 'SHA256SUMS.sha256'), 'Invalid seal path')
        seen.add(name)
        require(hash_file(root / name) == sha, 'Sealed content mismatch: ' + name)
    inv = inventory(root, ('SHA256SUMS', 'SHA256SUMS.sha256'))
    require(seen == {f['path'] for f in inv['files']}, 'Extra/missing sealed file')
    if (root / 'DIRECTORY_INVENTORY.json').exists():
        require(read_json(root / 'DIRECTORY_INVENTORY.json') == sorted(inv['directories']), 'Directory inventory mismatch')
    else:
        expected_dirs = {p.as_posix() for name in seen for p in Path(name).parents if p != Path('.')}
        require(set(inv['directories']) == expected_dirs, 'Extra/empty directory')
    for inventory_name in ('artifact_inventory.json','bundle_inventory.json'):
        if not (root / inventory_name).exists():continue
        recorded=read_json(root/inventory_name)
        if recorded['files'] and isinstance(recorded['files'][0],str):
            full=inventory(root)
            require(len(recorded['files'])==len(set(recorded['files'])) and
                set(recorded['files'])=={f['path'] for f in full['files']} and
                sorted(recorded['directories'])==sorted(full['directories']),'Upstream path inventory mismatch')
        else:
            actual=inventory(root,(inventory_name,'SHA256SUMS','SHA256SUMS.sha256'))
            require(set(recorded)=={'files','directories'} and
                sorted(recorded['files'],key=lambda f:f['path'])==actual['files'] and
                sorted(recorded['directories'])==actual['directories'],'Content inventory mismatch')
    return dict(seal=expected, payload_files_verified=len(seen), directories_verified=len(inv['directories']))


def seal_tree(root, inventory_name='bundle_inventory.json'):
    inv = inventory(root, (inventory_name, 'SHA256SUMS', 'SHA256SUMS.sha256'))
    write_json(root / inventory_name, inv)
    inv = inventory(root, ('SHA256SUMS', 'SHA256SUMS.sha256'))
    raw = ''.join(f"{f['sha256']}  {f['path']}\n" for f in inv['files']).encode()
    write_new(root / 'SHA256SUMS', raw)
    seal = digest(raw)
    write_new(root / 'SHA256SUMS.sha256', (seal + '  SHA256SUMS\n').encode())
    verify_seal(root, seal)
    return seal


def source_fingerprint():
    s = safe_path(SOURCE['path']).stat()
    return dict(bytes=s.st_size, device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')).decode().strip()


def git_state():
    return dict(HEAD=git('rev-parse', 'HEAD'), status=git('status', '--porcelain=v1', '--untracked-files=normal'),
                tracked=git('diff', '--name-only'), staged=git('diff', '--cached', '--name-only'))


def future_namespaces_absent():
    for path in (DATA, RUNS, PENDING):
        require(not safe_path(path).exists(), 'Future namespace already exists: ' + path.name)
    require(not list(REPORT.glob('.graph_development_data_v3.pending*')) and
            not list(REPORT.glob('.graph_development_runs_v3.pending*')), 'Unreviewed pending evidence exists')
    verify_v1_failure()
    for name in ('graph_development_data_v2','graph_development_runs_v2','.graph_development_data_v2.pending'):
        require(not safe_path(REPORT/name).exists(),'Historical v2 runtime cannot be reused')


def verify_v1_failure():
    """V1 is evidence only: exact authorization and empty units, never a reusable workspace."""
    safe_path(V1_PENDING)
    require(V1_PENDING.is_dir() and {p.name for p in V1_PENDING.iterdir()}=={'authorization.json','units'},
            'V1 failed evidence inventory differs')
    units=safe_path(V1_PENDING/'units');path=safe_path(V1_PENDING/'authorization.json')
    require(units.is_dir() and not list(units.iterdir()) and path.is_file(),'V1 failed units must remain empty')
    expected=dict(HEAD=HEAD,authorization='HUMAN_REVIEWED_EXPERIMENT_2B_FULL_TRAIN_GRAPH_DATA_V1',
        bundle_seal=V1_BUNDLE_SEAL,scope='Graph data only; model execution is forbidden')
    require(path.read_bytes()==canonical(expected),'V1 failed authorization bytes differ')
    for name in ('graph_development_data_v1','graph_development_runs_v1'):
        require(not safe_path(REPORT/name).exists(),'V1 runtime cannot be reused')
    def metadata(item):
        s=item.stat()
        return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,
                    attributes=getattr(s,'st_file_attributes',0))
    inv=inventory(V1_PENDING)
    return dict(path=str(V1_PENDING),inventory=inv,inventory_sha256=digest(canonical(inv)),
        authorization=expected,authorization_sha256=hash_file(path),units_count=0,
        metadata={'.':metadata(V1_PENDING),'units':metadata(units),'authorization.json':metadata(path)},
        preserved=True,reused=False,resumed=False,renamed=False,deleted=False)


def reject_n20_path(path):
    name = norm(path)
    require(not any(v in name for v in ('train20', 'expand20', 'n20', 'graph_fit_export_v1',
        'gat_paired', 'carla_final_', 'graph_partition_freeze_v1/../')),
        'Historical N20/final-evaluation path is NOT CURRENT')


def make_guard(phase='prepare', source_integrity=False):
    require(phase in ('prepare', 'synthetic', 'preflight', 'graph-data'), 'Unknown guard phase')
    from subprocess_policy import authenticated_policy, subprocess_admission
    policy=authenticated_policy()
    writable = (BUNDLE,) if phase in ('prepare', 'synthetic') else (DATA, PENDING) if phase == 'graph-data' else ()
    roots = tuple(norm(p) for p in writable)
    counters = dict(ZERO, blocked_TEST_attempts=0, blocked_network_attempts=0,
        blocked_historical_semantic_reads=0, blocked_writes=0, blocked_source_payload_reads=0,
        blocked_subprocess_attempts=0, blocked_model_imports=0,admitted_scientific_git_reads=[])
    current_prefix = norm(REPORT) + os.sep
    historical_reports = norm(REPO / 'reports') + os.sep
    source = norm(SOURCE['path'])
    protocol = norm(PROTOCOL)
    def check(path, write=False):
        if not isinstance(path, (str, bytes, os.PathLike)): return
        name = norm(path)
        if name == norm(os.devnull): return
        if any(p in ('test', 'test_normal', 'test_anomaly') for p in Path(name).parts) or \
                'carlanomaly-base-test' in name or Path(name).name == 'anomaly-observation.feather':
            counters['blocked_TEST_attempts'] += 1; require(False, 'TEST/official labels forbidden')
        if write:
            require(not name == source, 'TRAIN archive is read-only')
            if not any(name == root or name.startswith(root + os.sep) for root in roots):
                counters['blocked_writes'] += 1; require(False, 'Write outside new authorized namespace forbidden')
        elif name == source and not (phase == 'graph-data' or source_integrity and _INTEGRITY.get() == name):
            counters['blocked_source_payload_reads'] += 1; require(False, 'Real payload replay forbidden in preparation/preflight')
        elif name.startswith(historical_reports) and not name.startswith(current_prefix) and name != protocol and \
                _INTEGRITY.get() != name:
            counters['blocked_historical_semantic_reads'] += 1
            require(False, 'Historical results may be hashed for preservation only, never semantically read')
    def guard(event, args):
        if event.startswith('socket.') and event!='socket.gethostname':
            counters['blocked_network_attempts'] += 1; require(False, 'Network forbidden')
        if event == 'import' and (str(args[0]) == 'cognix' or str(args[0]).startswith('cognix.')):
            counters['blocked_model_imports'] += 1; require(False, 'Broad cognix import forbidden; authenticated definitions only')
        if event == 'import' and str(args[0]).split('.')[0] in ('torch', 'tensorflow', 'jax') and phase != 'synthetic':
            counters['blocked_model_imports'] += 1; require(False, 'Model imports only allowed for synthetic tests')
        if event == 'subprocess.Popen':
            admission=subprocess_admission(args,policy)
            if not admission:
                counters['blocked_subprocess_attempts'] += 1; require(False, 'Subprocess not approved')
            if admission['kind']=='authenticated-scientific-git-show':
                counters['admitted_scientific_git_reads'].append(admission)
        if event == 'open':
            mode, flags = args[1:3]
            check(args[0], (isinstance(mode,str) and any(c in mode for c in 'wax+')) or
                  bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.chmod','os.utime','os.link','os.symlink','shutil.copyfile'):
            for path in args[:2 if event in ('os.rename','os.link','os.symlink','shutil.copyfile') else 1]: check(path, True)
    return guard, counters


def install_guard(phase='prepare', source_integrity=False):
    global ACTIVE_PHASE
    require(ACTIVE_PHASE is None, 'Guard phase cannot be changed in a running process')
    ACTIVE_PHASE = phase
    guard, counters = make_guard(phase, source_integrity)
    sys.addaudithook(guard)
    return counters


def preservation_snapshot(progress=lambda message: None):
    roots = [REPO / 'reports', REPO / 'cognix', REPO / '.gitattributes', REPO / 'pyproject.toml']
    paths, directories = [], []
    for root in roots:
        safe_path(root)
        for path in [root, *sorted(root.rglob('*'))] if root.is_dir() else [root]:
            if path == BUNDLE or path.is_relative_to(BUNDLE) or path == DATA or path.is_relative_to(DATA) or \
                    path == PENDING or path.is_relative_to(PENDING) or path == RUNS or path.is_relative_to(RUNS): continue
            s=path.lstat()
            require(not (getattr(s,'st_file_attributes',0) & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0)), 'Reparse preservation entry')
            if stat.S_ISDIR(s.st_mode): directories.append(path.relative_to(REPO).as_posix())
            else:
                require(stat.S_ISREG(s.st_mode), 'Nonregular preservation file'); paths.append((path,s))
    progress('Opaque SHA256 preservation audit: %d files (all reports and scientific sources)' % len(paths))
    def record(item):
        path,s=item
        sha=hash_file(path,admitted=True)
        require(path.lstat().st_mtime_ns==s.st_mtime_ns, 'Preserved file changed')
        return dict(path=path.relative_to(REPO).as_posix(),bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                    ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino,sha256=sha)
    with ThreadPoolExecutor(max_workers=8) as pool: files=list(pool.map(record,paths))
    return dict(files=files,directories=directories,archive_fingerprint=source_fingerprint(),
                scope='All pre-existing reports and cognix files, including historical graph/final/conformal evidence; '
                      'opaque integrity hashing only; source archive identity metadata; read access times excluded')


def verify_preservation(baseline, progress=lambda message: None):
    current=preservation_snapshot(progress)
    require(current==baseline,'Historical evidence/source changed; do not repair')
    return dict(status='PASS',files_verified=len(current['files']),directories_verified=len(current['directories']),
                changed_files=[],added_files=[],removed_files=[],all_content_and_write_metadata_equal=True,
                baseline_sha256=digest(canonical(baseline)))
