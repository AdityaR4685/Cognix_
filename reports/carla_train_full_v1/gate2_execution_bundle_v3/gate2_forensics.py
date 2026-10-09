"""Independent, read-only binding of the stopped v1 engineering attempt."""
from gate2_common import (V1, V1_RUNTIME, V1_SEAL, HEAD, V8_SEAL, PARTITION_SHA,
    PARTITION_SEAL, SOURCE, V2, V2_RUNTIME, V2_SEAL, require, safe_path, read_json, verify_seal,
    hash_file, canonical, digest)


def verify_v1_failure():
    bundle = verify_seal(V1, V1_SEAL)
    root = safe_path(V1_RUNTIME)
    require({p.name for p in root.iterdir()} == {'run_manifest.json', 'lease', 'units', 'attempts'},
            'Unexpected v1 runtime/scientific result')
    require((root / 'units').is_dir() and not any((root / 'units').iterdir()) and
            (root / 'lease').read_bytes() == b'0', 'V1 committed state or lease changed')
    m = read_json(root / 'run_manifest.json')
    require((root / 'run_manifest.json').read_bytes() == canonical(m) and
            m['schema'] == 'transactional-Experiment-2B-Gate2-v1' and m['HEAD'] == HEAD and
            m['bundle_seal'] == V1_SEAL and m['v8_seal'] == V8_SEAL and
            m['partition_sha256'] == PARTITION_SHA and m['partition_runtime_seal'] == PARTITION_SEAL,
            'V1 runtime manifest binding mismatch')
    require(m['bindings'] == read_json(V1 / 'execution_bindings.json') and
            m['membership'] == m['bindings']['membership'] and
            m['environment'] == m['bindings']['environment'] and
            m['source_binding']['identity'] == SOURCE and
            m['upstream_completion']['source_binding'] == m['source_binding'] and
            m['TEST_requests'] == m['network_requests'] == 0 and
            m['graph_constructed'] is False and m['GAT_executed'] is False,
            'V1 source/scientific/activity mismatch')
    attempts = list((root / 'attempts').iterdir())
    require(len(attempts) == 1 and attempts[0].is_dir() and
            attempts[0].name == 'adf4914835fe4f7d95359aca0942a734', 'Ambiguous v1 attempt')
    attempt = safe_path(attempts[0])
    require({p.name for p in attempt.iterdir()} == {'attempt.json', 'failure_ledger.json',
            'methodology_decision.md', 'SHA256SUMS', 'SHA256SUMS.sha256'},
            'V1 pending or scientific result present')
    seal = verify_seal(attempt, 'fcc559944f14d9da63d500afd86ccd93e3e6bddf9da8722a60cb4d40d8bbe587')
    request = read_json(attempt / 'attempt.json')
    ledger = read_json(attempt / 'failure_ledger.json')
    require(request['attempt'] == attempt.name and
            request['run_manifest_sha256'] == digest(canonical(m)), 'V1 attempt manifest mismatch')
    require(ledger['phase'] == 'FIT_clean_only_Camera' and ledger['status'] == 'EXECUTION_STOPPED' and
            ledger['committed_units'] == {} and ledger['TEST_requests'] == ledger['network_requests'] == 0 and
            ledger['graph_constructed'] is False and ledger['GAT_executed'] is False and
            ledger['scientific_repair_performed'] is False and ledger['preserve_all_evidence'] is True,
            'V1 failure state mismatch')
    require(ledger['exception'] == "FileNotFoundError(2, 'No such file or directory')" and
            all(s in ledger['traceback'] for s in ('assemble_fit', 'load_clean', 'Town01__scenario-1',
                                                 'arrays.json', 'FileNotFoundError')),
            'V1 failure cause mismatch')
    executor = (V1 / 'execute_gate2.py').read_text()
    require(executor.index('x, binding = assemble_fit(') < executor.index('agent.fit(x)'),
            'V1 fitting order changed')
    return {'v1_bundle': str(V1), 'v1_bundle_seal': bundle['seal'], 'runtime': str(root),
        'run_manifest_sha256': hash_file(root / 'run_manifest.json'), 'attempt': attempt.name,
        'attempt_seal': seal['seal'], 'attempt_json_sha256': hash_file(attempt / 'attempt.json'),
        'failure_ledger_sha256': hash_file(attempt / 'failure_ledger.json'),
        'methodology_decision_sha256': hash_file(attempt / 'methodology_decision.md'),
        'failure_ledger': ledger, 'committed_units': {}, 'units_directory_empty': True,
        'scientific_result_obtained': False, 'failure_preceded_agent_fit': True,
        'real_fitting_executed': False, 'real_calibration_executed': False,
        'real_pseudo_generated': False, 'graph_constructed': False, 'GAT_executed': False,
        'TEST_requests': 0, 'network_requests': 0, 'historical_runtime_mutated': False,
        'action': 'STOP FOR HUMAN REVIEW'}


V2_FIT_SEALS = {
    'fit-Camera': '531ec474585318ac91ba8d7f281690c432a4a444fee96cf743a5a44222780bb6',
    'fit-IMU': '3ca96bf8a5aa1042619ab721c3f517bb38df0e592ff5f60aa509e946de4d3bd4',
    'fit-Seg': 'ee72ee058a38ca87c451e26f4c814d7db92a1ac000d6cbbe278ebcded22f8075'}


def verify_v2_failure():
    verify_seal(V2, V2_SEAL)
    root = safe_path(V2_RUNTIME)
    require({p.name for p in root.iterdir()} == {'run_manifest.json','lease','units','attempts'} and
            (root/'lease').read_bytes() == b'0', 'Unexpected v2 runtime or final scientific evidence')
    m = read_json(root/'run_manifest.json')
    require((root/'run_manifest.json').read_bytes() == canonical(m) and
            m['schema'] == 'transactional-Experiment-2B-Gate2-v2' and m['HEAD'] == HEAD and
            m['bundle_seal'] == V2_SEAL and m['v8_seal'] == V8_SEAL and
            m['partition_sha256'] == PARTITION_SHA and m['partition_runtime_seal'] == PARTITION_SEAL and
            m['bindings'] == read_json(V2/'execution_bindings.json') and
            m['membership'] == m['bindings']['membership'] and m['environment'] == m['bindings']['environment'] and
            m['source_binding']['identity'] == SOURCE and
            m['upstream_completion']['source_binding'] == m['source_binding'] and
            m['block_resolutions'] == read_json(V2/'verified_block_resolutions.json') and
            m['historical_v1_failure_binding'] == m['bindings']['v1_failure_evidence_sha256'] and
            m['graph_constructed'] is False and m['GAT_executed'] is False and
            m['TEST_requests'] == m['network_requests'] == 0, 'V2 runtime immutable binding mismatch')
    units = root/'units'
    require({p.name for p in units.iterdir()} == set(V2_FIT_SEALS), 'V2 unit set changed or CAL/audit/decision exists')
    fit_records = {}
    for name, expected in V2_FIT_SEALS.items():
        path = safe_path(units/name)
        verify_seal(path, expected)
        unit = read_json(path/'unit.json')
        fit = read_json(path/'fit_binding.json')
        require(unit['name'] == name and unit['run_manifest_sha256'] == hash_file(root/'run_manifest.json') and
                unit['inputs']['partition_sha256'] == PARTITION_SHA and
                unit['inputs']['fit_input_sha256'] == fit['input_array_sha256'] and
                fit['FIT_NORMAL'] == m['membership']['FIT_NORMAL'] and
                fit['CAL_NORMAL_excluded'] == m['membership']['CAL_NORMAL'] and
                fit['ordered_consumed_scenarios'] == sorted(m['membership']['FIT_NORMAL']) and
                fit['total_rows'] == 76*2999 and fit['bootstrap_members'] == 5 and fit['bootstrap_seed'] == 42 and
                fit['pseudo_rows'] == fit['TEST_rows'] == 0, 'V2 immutable FIT provenance mismatch')
        fit_records[name] = {'seal':expected,'fit_binding_sha256':hash_file(path/'fit_binding.json'),
            'unit_json_sha256':hash_file(path/'unit.json'),
            'upstream_state_hashes':read_json(path/'upstream_state_hashes.json')}
    attempts = list((root/'attempts').iterdir())
    require(len(attempts) == 1 and attempts[0].name == 'ea6a5f043e444d5badcd0d1fb999c4a5',
            'Ambiguous v2 stopped attempt')
    attempt = safe_path(attempts[0])
    attempt_seal = verify_seal(attempt)['seal']
    request = read_json(attempt/'attempt.json')
    ledger = read_json(attempt/'failure_ledger.json')
    require(request['attempt'] == attempt.name and
            request['run_manifest_sha256'] == hash_file(root/'run_manifest.json') and
            ledger['phase'] == 'CAL_pseudo_only' and ledger['status'] == 'EXECUTION_STOPPED' and
            ledger['exception'] == "KeyError('record_sha256')" and ledger['committed_units'] == V2_FIT_SEALS and
            ledger['graph_constructed'] is False and ledger['GAT_executed'] is False and
            ledger['TEST_requests'] == ledger['network_requests'] == 0 and
            ledger['scientific_repair_performed'] is False and ledger['preserve_all_evidence'] is True,
            'V2 stopped execution ledger mismatch')
    require("self.current['record_sha256']" in ledger['traceback'] and 'scenario_complete' in ledger['traceback'],
            'V2 evidence-binding failure cause changed')
    raw = safe_path(attempt/'raw-Town01__scenario-5')
    require(raw.is_dir() and {p.name for p in attempt.iterdir()} == {'attempt.json','failure_ledger.json',
            'methodology_decision.md','DIRECTORY_INVENTORY.json','SHA256SUMS','SHA256SUMS.sha256',raw.name},
            'V2 raw/pending/failure inventory changed')
    files = [p for p in raw.rglob('*') if p.is_file()]
    require(len(files) == 1288 and sum(p.stat().st_size for p in files) == 329415042,
            'V2 failed raw CAL workspace incomplete')
    owned = read_json(raw/'owned_workspace.json')
    require(owned['scenario_id'] == 'Town01/scenario-5' and
            owned['run_manifest_sha256'] == hash_file(root/'run_manifest.json'), 'V2 raw workspace ownership mismatch')
    return {'v2_bundle':str(V2),'v2_bundle_seal':V2_SEAL,'runtime':str(root),
        'run_manifest_sha256':hash_file(root/'run_manifest.json'),'attempt':attempt.name,
        'attempt_seal':attempt_seal,'failure_ledger_sha256':hash_file(attempt/'failure_ledger.json'),
        'attempt_json_sha256':hash_file(attempt/'attempt.json'),'failure_ledger':ledger,
        'committed_units':V2_FIT_SEALS,'FIT_records':fit_records,
        'no_CAL_source_or_audit_or_decision_unit':True,'FINAL_absent':True,
        'scientific_PASS_FAIL_result_obtained':False,'scientific_validity_inferred':False,
        'historical_FIT_units_are_intermediate_evidence_only':True,
        'raw_workspace':str(raw),'raw_files':len(files),'raw_bytes':sum(p.stat().st_size for p in files),
        'raw_workspace_bound_by_verified_attempt_seal':True,
        'v2_mutated_resumed_repaired_resealed_or_cleaned':False,
        'graph_constructed':False,'GAT_executed':False,'TEST_requests':0,'network_requests':0,
        'scientific_repair_performed':False,'action':'STOP FOR HUMAN REVIEW'}
