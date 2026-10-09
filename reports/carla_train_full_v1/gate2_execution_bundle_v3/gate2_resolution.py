"""Read-only storage amendment using the exact, authenticated sealed verifiers.

Production admission comes from frozen source inventory and partition identity.
No path in adoption.json is followed. The canonical adoption verifier constructs
the v7 path from the sealed historical proof, independently verifies that block,
and compares the entire adoption payload to its expected payload.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
from gate2_common import (BUNDLE, V8, V8_SEAL, V7, V7_SEAL, STORE, V7_STORE,
    PARTITION, PARTITION_SEAL, PARTITION_SHA, SOURCE, require, verify_seal,
    safe_path, read_json, hash_file, canonical, digest)

NATIVE = 'V8_NATIVE_EXTRACTED'
ADOPTED = 'V7_TO_V8_IDENTITY_ADOPTION'
_CONTEXT = None


def authenticate_verifiers():
    # Absolute legacy imports are safe only with a fully authenticated bundle
    # and no same-name module from a different directory already in sys.modules.
    verify_seal(V8, V8_SEAL)
    verify_seal(V7, V7_SEAL)
    for path in V8.glob('*.py'):
        module = sys.modules.get(path.stem)
        require(module is None or Path(getattr(module, '__file__', '')).resolve() == path,
                'Ambiguous sealed v8 import: ' + path.stem)
    if str(V8) not in sys.path:
        sys.path.append(str(V8))
    # Resolve every legacy module before executing any of its code. An earlier
    # sys.path directory must not be able to run a shadow module and only then
    # fail the post-import origin check.
    for path in V8.glob('*.py'):
        spec = importlib.util.find_spec(path.stem)
        require(spec is not None and spec.origin is not None and
                Path(spec.origin).resolve() == path,
                'Ambiguous sealed v8 discovery: ' + path.stem)
    modules = [importlib.import_module(name) for name in
               ('recovery_common', 'local_inventory', 'clean_block_store',
                'v7_clean_block_store', 'v7_adoption', 'local_source')]
    for module in modules:
        require(Path(module.__file__).resolve() == V8 / (module.__name__ + '.py'),
                'Wrong sealed verifier origin')
    require(modules[0].BUNDLE == V8 and modules[0].V7_ROOT == V7_STORE and
            modules[0].V7_SEAL == V7_SEAL, 'Sealed verifier root/seal binding mismatch')
    # These imports define functions/classes; no extraction/publication entry
    # point is invoked. Guarded tests also demonstrate zero upstream mutations.
    return modules


class ResolutionContext:
    """Frozen admission data; tests may supply artificial directories/data."""
    def __init__(self, store, old_store, scenarios, source, science, proof,
                 expected, verifiers, membership):
        self.store, self.old_store = safe_path(store), safe_path(old_store)
        self.scenarios = {r['scenario_id']: r for r in scenarios}
        self.source, self.science, self.proof = source, science, proof
        self.expected = {r['scenario_id']: r for r in expected}
        self.native, self.original, self.adoption = verifiers
        self.membership = membership
        require(len(self.scenarios) == len(scenarios) == len(self.expected) == len(expected),
                'Duplicate/ambiguous resolution admission')
        require(set(self.scenarios) == set(self.expected) ==
                set(membership['FIT_NORMAL'] + membership['CAL_NORMAL']) and
                not set(membership['FIT_NORMAL']) & set(membership['CAL_NORMAL']),
                'Resolution membership mismatch')
        require(Path(proof['v7_runtime_root']) == self.old_store,
                'Historical v7 root escape')
        self.identity_sha256 = digest(canonical(self.identity()))

    def identity(self):
        return {'store': str(self.store), 'old_store': str(self.old_store),
                'scenarios': self.scenarios, 'source': self.source,
                'science': self.science, 'proof': self.proof,
                'expected': self.expected, 'membership': self.membership}

    def unchanged(self):
        require(digest(canonical(self.identity())) == self.identity_sha256,
                'Resolution admission mutated')


def real_context():
    legacy, inventory, native, original, adoption, local_source = authenticate_verifiers()
    legacy.verify_science()
    verify_seal(PARTITION, PARTITION_SEAL)
    require(hash_file(PARTITION / 'partition.json') == PARTITION_SHA, 'Resolution partition SHA mismatch')
    partition = read_json(PARTITION / 'partition.json')
    bindings = read_json(BUNDLE / 'execution_bindings.json')
    membership = {role: partition[role] for role in ('FIT_NORMAL', 'CAL_NORMAL')}
    require(membership == bindings['membership'], 'Frozen resolution roles changed')
    receipt = read_json(V8 / 'source_verification.json')
    source = {k: receipt[k] for k in ('schema', 'identity', 'fingerprint', 'independently_measured_full_sha256')}
    require(source['identity'] == SOURCE and source['independently_measured_full_sha256'] == SOURCE['sha256'],
            'Resolution source binding mismatch')
    index = inventory.verify_inventory(V8, source)
    science = native.scientific_hashes()
    require(science == bindings['scientific_hashes'], 'Resolution scientific binding mismatch')
    proof = read_json(V8 / 'v7_failure_and_adoption_proof.json')
    require(Path(proof['v7_runtime_root']) == V7_STORE and proof['v7_seal_sha256'] == V7_SEAL,
            'Historical root/seal escape')
    adoption.audit_candidates(index, source, proof)
    expected = partition['v8_COMPLETE_GATE1']['completed_blocks']
    require(len(expected) == len(index['scenarios']) == 101 and
            partition['n_fit'] == len(membership['FIT_NORMAL']) == 76 and
            partition['n_cal'] == len(membership['CAL_NORMAL']) == 25,
            'Resolution counts changed')
    require({p.name for p in (STORE / 'verified_clean_blocks').iterdir()} ==
            {native.block_name(r['scenario_id']) for r in index['scenarios']},
            'Unexpected/missing v8 store block')
    return ResolutionContext(STORE, V7_STORE, index['scenarios'], source, science,
                             proof, expected, (native, original, adoption), membership)


def context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = real_context()
    return _CONTEXT


def snapshot(path, names):
    safe_path(path)
    require(path.is_dir() and {p.name for p in path.iterdir()} == set(names),
            'Incomplete/ambiguous block inventory')
    records = {}
    for name in names:
        item = safe_path(path / name)
        require(item.is_file(), 'Non-file block entry')
        s = item.stat()
        records[name] = {'sha256': hash_file(item), 'bytes': s.st_size,
                         'device': s.st_dev, 'inode': s.st_ino,
                         'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}
    return records


def resolve_arrays(sid, admitted=None):
    ctx = context() if admitted is None else admitted
    ctx.unchanged()
    require(sid in ctx.scenarios and sid in ctx.expected, 'Scenario outside frozen TRAIN membership')
    scenario, expected = ctx.scenarios[sid], ctx.expected[sid]
    ordinal = scenario['archive_ordinal']
    require(expected['scenario_id'] == sid and expected['archive_ordinal'] == ordinal and
            scenario['town'] == sid.split('/')[0], 'Resolution scenario/ordinal mismatch')
    # block_name rejects unsafe IDs before any feature path is constructed.
    path = safe_path(ctx.store / 'verified_clean_blocks' / ctx.native.block_name(sid))
    adopted = expected['kind'] == ADOPTED
    require(expected['kind'] in (ADOPTED, 'NEW_V8_EXTRACTION'), 'Unknown storage form')
    require(adopted == (1 <= ordinal <= ctx.proof['completed_scenarios']),
            'Storage kind/ordinal mismatch')
    names = ('adoption.json', 'oov_ledger.jsonl.gz') if adopted else \
            ('arrays.json', 'arrays.npz', 'manifest.json', 'oov_ledger.jsonl.gz')
    before = snapshot(path, names)
    if adopted:
        record = read_json(path / 'adoption.json')  # also rejects duplicate keys/NaN
        v7_path = safe_path(ctx.old_store / 'verified_clean_blocks' / ctx.native.block_name(sid))
        require(record['payload']['v7_block_path'] == str(v7_path), 'Adoption path escape/alias')
        old_before = snapshot(v7_path, ('arrays.json', 'arrays.npz', 'manifest.json'))
        read_json(v7_path / 'manifest.json')
        read_json(v7_path / 'arrays.json')
        verified = ctx.adoption.verify_adoption(path, scenario, ctx.source, ctx.science, ctx.proof)
        # verify_adoption already calls original.verify_block independently. A
        # second explicit call binds its readback into Gate-2 consumption evidence.
        old_verified = ctx.original.verify_block(v7_path, scenario, ctx.source,
                                                 ctx.proof['v7_scientific_hashes'])
        require(old_verified == ctx.proof['completed_blocks'][ordinal - 1] and
                verified['v7_block_path'] == str(v7_path), 'Verified v7 resolution mismatch')
        resolved = v7_path
    else:
        read_json(path / 'manifest.json')
        read_json(path / 'arrays.json')
        verified = ctx.native.verify_block(path, scenario, ctx.source, ctx.science)
        old_verified = None
        resolved = path
    require(verified == expected, 'Frozen block/adoption record changed')
    manifest = read_json(resolved / 'arrays.json')
    # Read only verified files; feature computation and publication are absent.
    with np.load(safe_path(resolved / 'arrays.npz'), allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)) and set(z.files) == ctx.original.ARRAY_KEYS,
                'Ambiguous NPZ keys')
        arrays = {name: z[name] for name in z.files}
    content = ctx.original.content_hash(arrays)
    require(content == verified['array_content_sha256'] == manifest['array_content_sha256'] and
            hash_file(resolved / 'arrays.npz') == verified['npz_sha256'],
            'Feature bytes changed after independent verification')
    require(snapshot(path, names) == before, 'v8 block changed during consumption')
    if adopted:
        require(snapshot(resolved, ('arrays.json', 'arrays.npz', 'manifest.json')) == old_before,
                'v7 block changed during consumption')
    ctx.unchanged()
    provenance = {'storage_form': ADOPTED if adopted else NATIVE, 'scenario_id': sid,
        'archive_ordinal': ordinal, 'v8_evidence_path': str(path),
        'resolved_feature_path': str(resolved), 'verified_v8_record': verified,
        'verified_v7_record': old_verified, 'v8_evidence_files': before,
        'resolved_feature_files': old_before if adopted else before,
        'admission_sha256': ctx.identity_sha256, 'v8_verifier_seal': V8_SEAL,
        'v7_verifier_seal': V7_SEAL, 'feature_recomputed': False,
        'adopted_feature_files_copied': False}
    return arrays, manifest, provenance


def verify_all_resolutions(admitted=None, reader=None):
    from gate2_data import load_clean
    ctx = context() if admitted is None else admitted
    reader = (lambda sid: load_clean(sid, admitted=ctx)) if reader is None else reader
    records = [reader(sid)['resolved_provenance'] for sid in sorted(ctx.scenarios)]
    counts = {form: sum(r['storage_form'] == form for r in records) for form in (NATIVE, ADOPTED)}
    role_counts = {role: {form: sum(r['storage_form'] == form and r['scenario_id'] in ids
                    for r in records) for form in (NATIVE, ADOPTED)}
                  for role, ids in ctx.membership.items()}
    return {'counts': counts, 'role_counts': role_counts, 'membership': ctx.membership,
            'resolutions': records, 'resolution_manifest_sha256': digest(canonical(records)),
            'feature_recomputed': False, 'adopted_feature_files_copied': False}
