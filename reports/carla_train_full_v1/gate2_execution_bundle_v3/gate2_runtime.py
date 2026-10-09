"""Exclusive OS lease, immutable sealed units, append-only attempts and final seal.

Windows byte-range locks are released by the OS even on process death. No stale
PID lock is deleted. Interrupted pending units stay as evidence and are never
adopted; deterministic work can be recomputed in a fresh attempt with the same
inputs, seeds and code. Committed units are always verified and reused.
"""
import datetime
import os
import traceback
import uuid
from contextlib import contextmanager
from gate2_common import (HEAD, V8_SEAL, PARTITION_SHA, PARTITION_SEAL, MODALITIES,
    require, safe_path, canonical, pretty, digest, read_json, write_new, verify_seal, seal_tree)
from gate2_audit import gate_decision


def manifest_for(bundle_seal, evidence):
    return {'schema': 'transactional-Experiment-2B-Gate2-v3', 'experiment': 'Experiment 2B',
            'HEAD': HEAD, 'bundle_seal': bundle_seal, 'v8_seal': V8_SEAL,
            'partition_sha256': PARTITION_SHA, 'partition_runtime_seal': PARTITION_SEAL,
            'bindings': evidence['bindings'], 'environment': evidence['environment'],
            'source_binding': evidence['source_binding'], 'membership': evidence['membership'],
            'upstream_completion': evidence['completion'],
            'block_resolutions': evidence.get('block_resolutions'),
            'replay_inventory_bindings': evidence.get('replay_inventory_bindings'),
            'historical_v1_failure_binding': evidence['bindings'].get('v1_failure_evidence_sha256'),
            'historical_v2_failure_binding': evidence['bindings'].get('v2_failure_evidence_sha256'),
            'FIT_policy': 'Recompute frozen FIT under v3 manifest; no cross-runtime adoption',
            'graph_constructed': False,
            'GAT_executed': False, 'TEST_requests': 0, 'network_requests': 0,
            'prediction_semantics': evidence['bindings']['prediction_semantics']}


def expected_units(membership):
    return {'fit-' + m for m in MODALITIES} | {'audit-' + m for m in MODALITIES} | \
        {'cal-' + sid.replace('/', '__') for sid in membership['CAL_NORMAL']} | {'cal-source-verification', 'decision'}


def verify_runtime(root, bundle_seal, evidence):
    root = safe_path(root)
    manifest = manifest_for(bundle_seal, evidence)
    require((root / 'run_manifest.json').read_bytes() == canonical(manifest), 'Conflicting runtime manifest')
    require((root / 'lease').read_bytes() == b'0', 'Lease evidence changed')
    allowed = {'run_manifest.json', 'lease', 'units', 'attempts', 'FINAL.json', 'SHA256SUMS',
               'SHA256SUMS.sha256', 'DIRECTORY_INVENTORY.json'}
    require({p.name for p in root.iterdir()} <= allowed and
            (root / 'units').is_dir() and (root / 'attempts').is_dir(), 'Foreign runtime state')
    valid = expected_units(evidence['membership'])
    committed = {}
    for unit in (root / 'units').iterdir():
        require(unit.is_dir() and unit.name in valid, 'Foreign/graph runtime unit')
        seal = verify_seal(unit)['seal']
        record = read_json(unit / 'unit.json')
        require(record['name'] == unit.name and record['run_manifest_sha256'] == digest(canonical(manifest)),
                'Committed unit binding mismatch')
        committed[unit.name] = seal
    for attempt in (root / 'attempts').iterdir():
        require(attempt.is_dir() and len(attempt.name) == 32 and
                all(c in '0123456789abcdef' for c in attempt.name), 'Foreign attempt')
        record = read_json(attempt / 'attempt.json')
        require(record['run_manifest_sha256'] == digest(canonical(manifest)) and
                record['attempt'] == attempt.name, 'Attempt binding mismatch')
        if (attempt / 'SHA256SUMS.sha256').exists():
            verify_seal(attempt)
    state = 'resumable_verified'
    if (root / 'FINAL.json').exists():
        require(set(committed) == valid, 'Premature final runtime')
        final = read_json(root / 'FINAL.json')
        statuses = {m: read_json(root / 'units' / ('audit-' + m) / 'audit.json')['status'] for m in MODALITIES}
        require(final == final_record(statuses, committed), 'Final runtime decision mismatch')
        if (root / 'SHA256SUMS.sha256').exists():
            verify_seal(root)
            state = 'complete_verified_immutable'
        else:
            state = 'final_seal_publication_pending'
    else:
        require(not (root / 'SHA256SUMS').exists() and not (root / 'SHA256SUMS.sha256').exists(),
                'Seal without final runtime')
    return {'state': state, 'committed': committed}


def final_record(statuses, committed):
    return {'status': gate_decision(statuses), 'modality_status': statuses,
            'committed_unit_seals': committed, 'graph_constructed': False, 'GAT_executed': False,
            'TEST_requests': 0, 'network_requests': 0, 'action': 'STOP FOR HUMAN REVIEW',
            'next_action': 'Separate human review; no automatic graph construction',
            'future_scientific_repair': 'Separately preregistered Experiment 2C or later'}


class Runtime:
    def __init__(self, root, bundle_seal, evidence):
        self.root, self.bundle_seal, self.evidence = safe_path(root), bundle_seal, evidence
        self.manifest = manifest_for(bundle_seal, evidence)
        if not root.exists():
            # A single target mkdir is the initialization claim. Partial initialization
            # fails closed; it is never overwritten or silently completed.
            root.mkdir()
            write_new(root / 'run_manifest.json', canonical(self.manifest))
            write_new(root / 'lease', b'0')
            (root / 'units').mkdir()
            (root / 'attempts').mkdir()
        self.verify()
        self.attempt = None

    def verify(self):
        return verify_runtime(self.root, self.bundle_seal, self.evidence)

    @contextmanager
    def lease(self):
        require(not (self.root / 'SHA256SUMS.sha256').exists(), 'Immutable completed runtime')
        with (self.root / 'lease').open('r+b') as stream:
            # Lock beyond EOF so independent readback/hash handles can still
            # read the immutable lease bytes while this process owns the lease.
            lock_offset = 1 << 30
            stream.seek(lock_offset)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            try:
                self.verify()
                yield self
            finally:
                stream.seek(lock_offset)
                unlock()

    def begin_attempt(self):
        require(not (self.root / 'FINAL.json').exists(), 'Final runtime cannot start new work')
        self.attempt = self.root / 'attempts' / uuid.uuid4().hex
        self.attempt.mkdir()
        write_new(self.attempt / 'attempt.json', canonical({
            'attempt': self.attempt.name, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_manifest_sha256': digest(canonical(self.manifest)),
            'resume_policy': 'Reuse only verified committed units; retain all orphan/pending evidence; same inputs/code/seeds'}))

    def has(self, name):
        return name in self.verify()['committed']

    def unit(self, name):
        require(name in self.verify()['committed'], 'Missing immutable unit')
        return self.root / 'units' / name

    def commit(self, name, writer, inputs):
        require(not (self.root / 'FINAL.json').exists() and self.attempt is not None,
                'No active mutable attempt')
        require(name in expected_units(self.evidence['membership']), 'Forbidden runtime unit')
        if self.has(name):
            existing = self.unit(name)
            require(read_json(existing / 'unit.json')['inputs'] == inputs, 'Resume input mismatch')
            return existing
        pending = self.attempt / ('pending-' + name)
        pending.mkdir()
        write_new(pending / 'unit_request.json', canonical({'name': name, 'inputs': inputs}))
        writer(pending)
        write_new(pending / 'unit.json', canonical({'name': name, 'inputs': inputs,
                  'run_manifest_sha256': digest(canonical(self.manifest))}))
        seal_tree(pending)
        destination = self.root / 'units' / name
        require(not destination.exists(), 'Concurrent unit publication')
        require(os.name == 'nt', 'Production publication is bound to Windows no-replace directory rename')
        os.rename(pending, destination)
        verify_seal(destination)
        return destination

    def finish_attempt(self, status, phase, exc=None):
        require(self.attempt is not None, 'No attempt')
        ledger = {'status': status, 'phase': phase, 'exception': repr(exc) if exc else None,
                  'traceback': traceback.format_exc() if exc else None,
                  'committed_units': self.verify()['committed'],
                  'preserve_all_evidence': True, 'scientific_repair_performed': False,
                  'graph_constructed': False, 'GAT_executed': False,
                  'TEST_requests': 0, 'network_requests': 0, 'action': 'STOP FOR HUMAN REVIEW'}
        write_new(self.attempt / ('failure_ledger.json' if exc else 'completion.json'), pretty(ledger))
        if exc:
            text = ('# Experiment 2B Gate-2 interrupted methodology decision\n\n'
                    'Execution stopped at ' + phase + '.\n\n' + repr(exc) + '\n\n'
                    'The full scientific-health evidence is incomplete. No Gate-2 PASS is claimed. '
                    'Preserve the failure ledger, traceback, committed seals and all pending/raw evidence. '
                    'Verified committed units may be reused by a separately human-authorized invocation with identical inputs, code and seeds. '
                    'There is no automatic scientific repair. Any new scientific repair requires separately preregistered Experiment 2C or later. '
                    'No graph construction or training may follow this stop.\n\nSTOP FOR HUMAN REVIEW\n')
            write_new(self.attempt / 'methodology_decision.md', text.encode())
        seal_tree(self.attempt)

    def finalize(self, statuses):
        committed = self.verify()['committed']
        require(set(committed) == expected_units(self.evidence['membership']), 'Incomplete Gate-2 evidence')
        raw = canonical(final_record(statuses, committed))
        if (self.root / 'FINAL.json').exists():
            require((self.root / 'FINAL.json').read_bytes() == raw, 'Final evidence mismatch')
        else:
            write_new(self.root / 'FINAL.json', raw)
        seal = seal_tree(self.root)
        self.verify()
        return dict(final_record(statuses, committed), runtime_seal=seal)
