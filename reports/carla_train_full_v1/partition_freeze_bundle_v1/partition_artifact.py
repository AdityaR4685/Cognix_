"""Canonical partition records and exclusive, evidence-preserving publication."""
import datetime
import os
import uuid
from pathlib import Path
from partition_common import (BUNDLE, V8, require, canonical_bytes, sha_bytes, hash_file,
    read_json, write_new, safe_path, verify_manifest)
from partition_algorithm import compute_partition, town_counts

MEMBER_FILE = 'member_evidence.jsonl.gz'
PARTITION_FILE = 'partition.json'
RUNTIME_FILES = {PARTITION_FILE, 'partition.json.sha256', MEMBER_FILE,
                 'DATA_INTEGRITY_GATE_PASS.json', 'SHA256SUMS', 'SHA256SUMS.sha256'}


def validate_timestamp(value):
    require(isinstance(value, str), 'Missing freeze timestamp')
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        require(False, 'Invalid freeze timestamp')
    require(parsed.tzinfo == datetime.timezone.utc and parsed.isoformat() == value, 'Noncanonical UTC timestamp')


def build_record(evidence, split, created_utc, bundle_seal):
    validate_timestamp(created_utc)
    by_id = {r['scenario_id']: r for r in evidence['completed']}
    scenarios = []
    for scenario in evidence['index']['scenarios']:
        sid = scenario['scenario_id']
        scenarios.append({
            'scenario_id': sid, 'archive_ordinal': scenario['archive_ordinal'],
            'town': scenario['town'], 'rows': scenario['emitted_rows'],
            'inventory_record_sha256': scenario['record_sha256'],
            'source_member_set_sha256': scenario['source_member_set_sha256'],
            'parser_member_chain_sha256': scenario['parser_member_chain_sha256'],
            'member_count': scenario['member_count'],
            'exact_member_hashes': {'file': MEMBER_FILE, 'scenario_id': sid,
                'record_sha256': scenario['record_sha256'],
                'encoding': 'exact sealed v8 gzip JSONL inventory bytes; includes every member path/size/SHA256 and TAR offsets'},
            'v8_block_or_adoption_identity': by_id[sid],
            'descriptors': evidence['descriptors'][sid]})
    return {'schema': 'immutable-Experiment-2B-full-TRAIN-partition-v1',
            'experiment': 'Experiment 2B', **split,
            'split_rule_version': {'source_schema_version': evidence['protocol']['schema_version'],
                'original_experiment': evidence['protocol']['experiment'],
                'path': 'reports/carla_train_full_v1/full_train_preregistered_protocol.json',
                'sha256': next(r['sha256'] for r in evidence['bindings']['protocols']
                               if r['name'] == 'full_train_preregistered_protocol.json')},
            'exact_split_rule': evidence['protocol']['split'],
            'town_counts': {'overall': town_counts(split['sorted_canonical_ids']),
                'CAL_NORMAL': town_counts(split['CAL_NORMAL']), 'FIT_NORMAL': town_counts(split['FIT_NORMAL'])},
            'scenario_records_in_archive_order': scenarios,
            'exact_member_evidence': {'file': MEMBER_FILE,
                'sha256': evidence['index']['ledger_sha256'],
                'logical_sha256': evidence['index']['ledger_logical_sha256']},
            'upstream_bindings': evidence['bindings'],
            'v8_COMPLETE_GATE1': evidence['completion'],
            'scientific_source_bindings': evidence['science'],
            'creation_provenance': {'created_utc': created_utc, 'created_by': 'partition_freeze.py',
                'preparation_bundle_seal_sha256': bundle_seal,
                'membership_inputs': 'accepted complete canonical TRAIN scenario IDs only',
                'encoding': "UTF-8 JSON; sort_keys=True; separators=(',', ':'); allow_nan=False",
                'historical_membership_reused': False, 'balancing_performed': False,
                'source_evidence_mutated': False},
            'immutable_after_successful_freeze': True, 'fitting_executed': False,
            'pseudo_rows': 0, 'GAT_executed': False, 'graph_constructed': False,
            'calibration_executed': False, 'TEST_requests': 0, 'network_requests': 0}


def gate_record(partition_sha, bundle_seal):
    return {'schema': 'Experiment-2B-DATA-INTEGRITY-GATE-v1', 'experiment': 'Experiment 2B',
            'status': 'DATA_INTEGRITY_GATE_PASS', 'partition_sha256': partition_sha,
            'preparation_bundle_seal_sha256': bundle_seal,
            'gate_spec': 'reports/carla_train_full_v1/development_gate_spec.json',
            'all_required_integrity_predicates_verified': True,
            'scope': 'DATA / INTEGRITY only; no model-performance criterion',
            'fitting_executed': False, 'pseudo_rows': 0, 'GAT_executed': False,
            'TEST_requests': 0, 'network_requests': 0,
            'Gate_2_authorization': 'Separate human authorization required after partition review',
            'future_scientific_repair': 'Separately preregister Experiment 2C or later; never silently amend Experiment 2B',
            'action': 'STOP FOR HUMAN REVIEW'}


def verify_partition_bytes(raw, expected, detached):
    expected_raw = canonical_bytes(expected)
    digest = sha_bytes(expected_raw)
    require(raw == expected_raw, 'Existing partition byte/semantic mismatch; overwrite forbidden')
    require(detached == (digest + '  partition.json\n').encode(), 'Detached partition SHA mismatch')
    return digest


def verify_existing(target, evidence, bundle_seal, split=None):
    target = safe_path(target)
    require(target.is_dir() and {p.name for p in target.iterdir()} == RUNTIME_FILES,
            'Existing partition incomplete or unexpected files; overwrite forbidden')
    verify_manifest(target)
    existing = read_json(target / PARTITION_FILE)
    timestamp = existing['creation_provenance']['created_utc']
    # One computation per invocation; an already computed freeze split is passed in.
    split = split if split is not None else compute_partition([r['scenario_id'] for r in evidence['completed']])
    expected = build_record(evidence, split, timestamp, bundle_seal)
    digest = verify_partition_bytes((target / PARTITION_FILE).read_bytes(), expected,
                                   (target / 'partition.json.sha256').read_bytes())
    require(hash_file(target / MEMBER_FILE) == evidence['index']['ledger_sha256'], 'Member evidence SHA mismatch')
    require((target / 'DATA_INTEGRITY_GATE_PASS.json').read_bytes() == canonical_bytes(gate_record(digest, bundle_seal)),
            'DATA/INTEGRITY PASS record mismatch')
    return {'status': 'VERIFIED_EXISTING_IMMUTABLE_PARTITION', 'partition_sha256': digest,
            'runtime_seal_sha256': hash_file(target / 'SHA256SUMS'), 'action': 'STOP FOR HUMAN REVIEW'}


def check_target_namespace(target):
    target = safe_path(target)
    require(not list(target.parent.glob('.' + target.name + '.pending-*')),
            'Pending prior partition publication evidence exists; no repair or cleanup')
    return target


def freeze(evidence, target, bundle_seal):
    target = check_target_namespace(target)
    split = compute_partition([r['scenario_id'] for r in evidence['completed']])
    if target.exists():
        return verify_existing(target, evidence, bundle_seal, split)
    created_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = build_record(evidence, split, created_utc, bundle_seal)
    raw = canonical_bytes(record)
    digest = sha_bytes(raw)
    pending = target.with_name('.' + target.name + '.pending-' + uuid.uuid4().hex)
    pending.mkdir()
    # All files are exclusive creations. A failure leaves the entire pending evidence.
    write_new(pending / PARTITION_FILE, raw)
    write_new(pending / 'partition.json.sha256', (digest + '  partition.json\n').encode())
    with (V8 / 'inventory_scenarios.jsonl.gz').open('rb') as source, (pending / MEMBER_FILE).open('xb') as output:
        for data in iter(lambda: source.read(8 << 20), b''):
            require(output.write(data) == len(data), 'Short member evidence copy')
        output.flush()
        os.fsync(output.fileno())
    write_new(pending / 'DATA_INTEGRITY_GATE_PASS.json', canonical_bytes(gate_record(digest, bundle_seal)))
    listing = ''.join(hash_file(p) + '  ' + p.name + '\n' for p in sorted(pending.iterdir())).encode()
    write_new(pending / 'SHA256SUMS', listing)
    write_new(pending / 'SHA256SUMS.sha256', (sha_bytes(listing) + '  SHA256SUMS\n').encode())
    verify_existing(pending, evidence, bundle_seal, split)
    require(not target.exists(), 'Concurrent partition target appeared; preserve pending evidence')
    # Windows os.rename refuses ANY existing target; never use replace.
    require(os.name == 'nt', 'This sealed publication adapter is bound to Windows no-replace directory rename')
    os.rename(pending, target)
    result = verify_existing(target, evidence, bundle_seal, split)
    result['status'] = 'DATA_INTEGRITY_GATE_PASS'
    return result
