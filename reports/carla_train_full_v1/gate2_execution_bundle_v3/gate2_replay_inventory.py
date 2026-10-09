"""Canonical raw-ledger -> sealed-index evidence binding, without feature work."""
import copy
import re
from pathlib import Path
from gate2_common import (V8, V8_SEAL, require, safe_path, read_json, hash_file,
                         canonical, digest, verify_seal)

RAW_SCHEMA = 'frozen-local-TRAIN-scenario-v7'
INDEX_SCHEMA = 'frozen-local-TRAIN-inventory-v7'
LEDGER_NAME = 'inventory_scenarios.jsonl.gz'


class ReplayInventory:
    """Production receives an independently verified index; tests use fake data.

    Hash/summary functions are the exact sealed local_inventory implementation.
    A raw record never receives or supplies an authoritative record_sha256.
    """
    def __init__(self, index, source_binding, legacy, files=()):
        self.legacy = legacy
        self.index = copy.deepcopy(index)
        self.source_binding = copy.deepcopy(source_binding)
        self.files = tuple((safe_path(path), sha) for path,sha in files)
        self.identity = digest(canonical({'index':self.index,'source':self.source_binding}))
        require(self.index.get('schema') == INDEX_SCHEMA and self.index.get('source_binding') == source_binding and
                self.index.get('ledger_name') == LEDGER_NAME, 'Wrong replay inventory schema/source/path')
        entries = self.index.get('scenarios')
        count = self.index.get('scenario_count')
        require(type(count) is int and count > 0 and isinstance(entries,list) and len(entries) == count,
                'Missing/extra replay index entry')
        seen = set()
        required = {'schema','scenario_id','archive_ordinal','first_tar_offset',
                    'source_member_set_sha256','parser_member_chain_sha256','n_ticks','emitted_rows','record_sha256'}
        for ordinal,entry in enumerate(entries,1):
            require(isinstance(entry,dict) and required <= set(entry) and
                    entry['schema'] == RAW_SCHEMA and type(entry['archive_ordinal']) is int and
                    entry['archive_ordinal'] == ordinal and isinstance(entry['scenario_id'],str) and
                    entry['scenario_id'] not in seen and
                    not {'members','metadata_integrity','image_readability'} & set(entry),
                    'Duplicate/unaligned or malformed replay scenario index')
            for key in ('record_sha256','source_member_set_sha256','parser_member_chain_sha256'):
                require(isinstance(entry[key],str) and re.fullmatch('[0-9a-f]{64}',entry[key]) is not None,
                        'Malformed replay inventory hash')
            require(type(entry['first_tar_offset']) is int and entry['first_tar_offset'] >= 0 and
                    type(entry['n_ticks']) is int and entry['n_ticks'] > 1 and
                    type(entry['emitted_rows']) is int and entry['emitted_rows'] == entry['n_ticks']-1,
                    'Malformed replay offset/row count')
            seen.add(entry['scenario_id'])
        require(self.index.get('scenario_index_sha256') == legacy.digest(entries),
                'Replay scenario-index digest mismatch')
        self.unchanged()

    def unchanged(self):
        require(self.identity == digest(canonical({'index':self.index,'source':self.source_binding})),
                'Replay index admission changed')
        for path,sha in self.files:
            require(safe_path(path).is_file() and hash_file(path) == sha,
                    'Sealed replay inventory ledger/index/seal changed')

    def bind(self, raw_full_scenario_record, expected_ordinal):
        self.unchanged()
        raw = raw_full_scenario_record
        require(isinstance(raw,dict) and 'record_sha256' not in raw and raw.get('schema') == RAW_SCHEMA,
                'Raw replay ledger must use canonical schema without caller-provided record_sha256')
        ordinal = raw.get('archive_ordinal')
        require(type(ordinal) is int and type(expected_ordinal) is int and
                ordinal == expected_ordinal and 1 <= ordinal <= self.index['scenario_count'],
                'Duplicate/unaligned replay scenario or missing index entry')
        # Exact existing upstream representation, including all summary fields.
        record_sha256 = self.legacy.digest(raw)
        reconstructed = dict(self.legacy.summary(raw), record_sha256=record_sha256)
        require(reconstructed == self.index['scenarios'][ordinal-1],
                'Canonical replay ledger/index scenario binding mismatch')
        return record_sha256

    def finish(self, ordinals):
        self.unchanged()
        require(ordinals == list(range(1,self.index['scenario_count']+1)),
                'Missing/extra/unaligned replay index coverage')


def load_verified_replay_inventory(source_binding):
    from gate2_resolution import authenticate_verifiers
    _,legacy,_,_,_,_ = authenticate_verifiers()
    verify_seal(V8,V8_SEAL)
    # Fixed paths are admitted before the canonical verifier reads anything.
    index_path = safe_path(V8/'source_inventory.json')
    ledger_path = safe_path(V8/LEDGER_NAME)
    strict_index = read_json(index_path)
    require(strict_index.get('ledger_name') == LEDGER_NAME, 'Replay ledger path escape')
    verified_index = legacy.verify_inventory(V8,source_binding)
    require(strict_index == verified_index, 'Ambiguous replay index parsing')
    files = [(p,hash_file(p)) for p in (index_path,ledger_path,V8/'SHA256SUMS',V8/'SHA256SUMS.sha256')]
    return ReplayInventory(verified_index,source_binding,legacy,files)


def verify_all_replay_bindings(source_binding):
    inventory = load_verified_replay_inventory(source_binding)
    records = []
    for ordinal,raw in enumerate(inventory.legacy.inventory_records(V8,inventory.index),1):
        verified_sha = inventory.bind(raw,ordinal)
        records.append({'scenario_id':raw['scenario_id'],'archive_ordinal':ordinal,
            'raw_ledger_contains_record_sha256':False,'canonical_record_sha256':verified_sha,
            'indexed_record_sha256':inventory.index['scenarios'][ordinal-1]['record_sha256'],
            'exact_canonical_summary_plus_digest_matches_index':True})
    inventory.finish([r['archive_ordinal'] for r in records])
    return {'schema':'Gate2-v3-canonical-replay-inventory-binding-proof',
        'scenario_count':len(records),'bindings':records,'index_schema':INDEX_SCHEMA,
        'index_file_sha256':hash_file(V8/'source_inventory.json'),
        'ledger_file_sha256':hash_file(V8/LEDGER_NAME),'v8_bundle_seal':V8_SEAL,
        'canonical_digest_function':'sealed local_inventory.digest(raw_full_scenario_record)',
        'canonical_summary_function':'sealed local_inventory.summary(raw_full_scenario_record)',
        'raw_records_mutated':False,'caller_provided_record_sha256_trusted':False,
        'real_fitting_executed':False,'real_calibration_executed':False,'real_pseudo_generated':False,
        'graph_constructed':False,'GAT_executed':False,'TEST_requests':0,'network_requests':0}
