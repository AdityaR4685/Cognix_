"""V3 preparation/sealing only; never run graph-data or real model entrypoints."""
import argparse
import datetime
from graph_common import *
from amendment_evidence import scientific_invariance,verify_amendment,write_diff
from execution_closure import pre_source_closure,complete_source_integrity,PRE_SOURCE_STEPS
from production_audits import segmentation_compatibility_audit
from subprocess_policy import policy_report

REQUIRED=('README.md','v1_failure_evidence.json','v2_review_findings.json','known_failure_recurrence_matrix.json',
    'engineering_amendment.json','subprocess_allowlist_audit.json','execution_path_closure_audit.json',
    'production_clean_resolution_audit.json','production_inventory_schema_audit.json','segmentation_compatibility_audit.json',
    'disk_resource_audit.json','execution_order_audit.json','scientific_invariance_audit.json','source_code_audit.json',
    'preservation_baseline.json','preservation_audit.json','synthetic_test_results.json',
    'production_integration_test_results.json','adversarial_test_results.json','future_commands.json',
    'bundle_inventory.json','SHA256SUMS','SHA256SUMS.sha256')


def initial():
    counters=install_guard('prepare',source_integrity=True)
    future_namespaces_absent();verify_seal(V1_BUNDLE,V1_BUNDLE_SEAL);verify_seal(V2_BUNDLE,V2_BUNDLE_SEAL)
    require(not (BUNDLE/'preparation_initial.json').exists(),'V3 initialization already exists')
    state=git_state();require(state['HEAD']==HEAD and not state['tracked'] and not state['staged'],'Git mismatch')
    baseline=preservation_snapshot(lambda message:print(message,flush=True))
    write_json(BUNDLE/'preservation_baseline.json',baseline)
    write_json(BUNDLE/'preparation_initial.json',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_state=state,source_fingerprint=source_fingerprint(),v1_pending=verify_v1_failure(),
        v1_bundle_seal=V1_BUNDLE_SEAL,v2_bundle_seal=V2_BUNDLE_SEAL,full_archive_hash_invoked=False))
    write_json(BUNDLE/'scientific_invariance_audit.json',scientific_invariance())
    require(canonical(policy_report())==canonical(read_json(V2_BUNDLE/'subprocess_allowlist_audit.json')),
        'Reviewed V2 subprocess policy semantics changed')
    write_new(BUNDLE/'subprocess_allowlist_audit.json',(V2_BUNDLE/'subprocess_allowlist_audit.json').read_bytes())
    v2_files=inventory(V2_BUNDLE)
    write_json(BUNDLE/'v2_review_findings.json',dict(status='ENGINEERING_PRODUCTION_REALISM_GATES_REQUIRED',
        v2_bundle_seal=V2_BUNDLE_SEAL,v2_valid_historical_evidence=True,
        retained_v2_conclusions=['80 tests PASS','scientific invariance PASS','real_context authentication PASS',
            'Gate1 verify_science and all 22 sealed Git reads PASS','arbitrary subprocess/Git/network blocked','V1 pending protected'],
        findings=['V2 checks resolution manifest and real_context but does not production-load all101 clean blocks',
            'V2 does not independently bind all101 raw production records through ReplayInventory and sealed summary/digest',
            'V2 full-source hash precedes predictable runtime closure checks'],
        scientific_failure=False,v2_files_modified=[],exact_v2_files_preserved=v2_files,
        evidence_provenance='Human review request; preserved sealed V2 source/tests',activity=dict(ZERO)))
    write_json(BUNDLE/'known_failure_recurrence_matrix.json',dict(status='ENGINEERING_REGRESSION_GATES_DEFINED',rows=[
        dict(failure='Graph data v1 authenticated git show blocked',prevention='Byte-identical reviewed V2 subprocess policy; actual22 reads and real_context under same future admission',
            evidence='subprocess_allowlist_audit.json; production_integration_test_results.json'),
        dict(failure='Gate2 v1 assumes adopted v7 block has physical v8 arrays',prevention='All101 canonical gate2_data.load_clean calls; exact33native/68adopted and sealed provenance',
            evidence='production_clean_resolution_audit.json'),
        dict(failure='Gate2 v2 fixture-only raw record_sha256 assumption',prevention='All101 actual raw records; exact sealed summary/digest/ReplayInventory.bind; inventedfield rejected',
            evidence='production_inventory_schema_audit.json'),
        dict(failure='Predictable resolver/schema/auth/environment/disk error after146GBhash',prevention='Shared fourteen pre-source gates; six fault injections prove zero hash/pending calls',
            evidence='execution_order_audit.json; adversarial_test_results.json')],
        historical_failures_unchanged=True,scientific_rescue_or_result_tuning=False))
    python=read_json(BUNDLE/'current_experiment_bindings.json')['environment']['python_executable']
    write_json(BUNDLE/'future_commands.json',dict(shell='Windows cmd.exe',
        read_only_preflight=f'"{python}" -B "{BUNDLE / "preflight_graph.py"}" --bundle-seal EXTERNALLY_REVIEWED_V3_SEAL',
        future_graph_data_generation=f'"{python}" -B "{BUNDLE / "generate_graph_data.py"}" --bundle-seal EXTERNALLY_REVIEWED_V3_SEAL --authorize-graph-data-generation {DATA_TOKEN}',
        external_seal_required=True,graph_data_command_executed=False,real_data_authorized=False,real_training_authorized=False,
        data=str(DATA),pending=str(PENDING),runs=str(RUNS)))
    write_json(BUNDLE/'engineering_amendment.json',dict(status='PREPARATION_ONLY_STOP_FOR_HUMAN_REVIEW',
        reason='Close mixed production clean-resolution and actual raw inventory schema gates before expensive source hash',
        v1_bundle_seal=V1_BUNDLE_SEAL,v2_bundle_seal=V2_BUNDLE_SEAL,science_changed=False,
        scientific_invariance='PASS',subprocess_policy_byte_identical_to_v2=True,
        pre_source_steps=list(PRE_SOURCE_STEPS),after_pre_source=['fresh_full_SOURCE_SHA','source_fingerprint_recheck',
            'create_new_V3_pending','record_V3_authorization','authenticated_source_replay','complete_replay_validation','publish_final_V3_data'],
        future_namespace=dict(data=str(DATA),pending=str(PENDING),runs=str(RUNS)),authorization_token=DATA_TOKEN,
        shared_closure='execution_closure.pre_source_closure',resolver_logic_reimplemented=False,
        inventory_digest_reimplemented=False,real_source_semantic_replay=False,real_model_execution=False,
        known_failures='known_failure_recurrence_matrix.json',activity=dict(ZERO)))
    readme=f'''# Experiment-2B graph development bundle v3

Preparation only. STOP FOR HUMAN REVIEW. No graph-data v3 or real model execution is authorized or executed.

V3 closes the two production-realism gaps identified in review of V2: all 101 clean blocks are read through authenticated gate2_data.load_clean / gate2_resolution.resolve_arrays, and all 101 raw records are read through the sealed inventory API and bound using its exact summary/digest and Gate2-v3 ReplayInventory.bind. This does not revisit V2's valid Git/authentication and scientific invariance conclusions.

V1 ({V1_BUNDLE_SEAL}) and V2 ({V2_BUNDLE_SEAL}) remain immutable. V1 failed pending evidence remains exactly its 260-byte authorization.json and empty units/. Its original read-only reproduction and evidence are copied byte-identically from V2, never rerun or resumed. The full preservation baseline binds every existing report/scientific file and write metadata. V2's exact preserved file inventory is in v2_review_findings.json.

No graph science changes. Exporter, graph replay, models, trainer, current restoration and upstream scientific definitions remain byte-identical to V2. The subprocess_policy.py file and its policy audit are byte-identical to V2. Only the exact sealed22 scientific Git reads, existing read-only Git forms and reviewed local Windows version probe remain admitted. No arbitrary shell, Git write/network, TEST or historical namespace write capability is added.

The SAME execution_closure.pre_source_closure serves preparation, preflight and future data. Fourteen mandatory steps check external seal, Git, historical preservation, absent future namespaces, upstream bindings, real_context, all101 clean blocks (33native/68adopted), all101 canonical raw inventory records, scientific restoration prerequisites without creating real objects, non-model imports, exact installed environment, disk/resources and barriers. Only complete_source_integrity may then read the full146453559283-byte archive for SHA verification. Source fingerprint is rechecked before pending creation. No sensor payload is parsed by preparation/preflight, clean features are never recomputed/copied, and models are never fitted/calibrated/scored in production checks.

Seg compatibility tests exercise the exact accepted sanitizer on synthetic IDs0..255. IDs0..28 are preserved;29..255 map to22. A runtime generate_pair probe stops immediately before the actual Seg pseudo implementation and proves sanitizer-before-recipe order. Seg29 and all pseudo semantics remain unchanged;31..255 have no added semantic interpretation.

The conservative disk gate preserves20GiB raw reserve, the128MiB retained-member cap, whole one-scenario scratch upper bound, two full artifact size budgets and8GiB additional margin. It deletes no historical evidence. Estimates and actual production maxima are in disk_resource_audit.json.

All original80 tests are retained (bounded synthetic CPU model fixtures only), with actual production read-only integration and ordering/fault-injection tests. Six simulated failures must stop before the full archive hash and pending creation. They alter no source/evidence. Source hashes in accepted results must match all sealed Python files.

Graph validation remains development-only: upstream fitting used all76 FIT including15 GRAPH_VAL. P(target=normal) concerns the TRAIN-derived constructed clean/pseudo task and is never a physical safety probability. No seed replacement, performance minimum or requirement that EpistemicGAT wins is introduced.

Future paths are {PENDING}, {DATA}, {RUNS}; V1 and V2 runtime namespaces cannot be written. No V3 runtime exists now. Every named requested audit is emitted separately and sealed with exact inventory and SHA256SUMS/detached external seal. Files become read-only. Commands in future_commands.json use cmd.exe syntax and require the external V3 seal reported at completion. The future data command must await separate human authorization.
'''
    write_new(BUNDLE/'README.md',readme.encode())
    closure=pre_source_closure(None,preparing=True,progress=lambda message:print(message,flush=True))
    write_json(BUNDLE/'execution_path_closure_audit.json',closure)
    write_json(BUNDLE/'production_clean_resolution_audit.json',closure['production_clean_resolution'])
    write_json(BUNDLE/'production_inventory_schema_audit.json',closure['production_inventory_schema'])
    write_json(BUNDLE/'disk_resource_audit.json',closure['disk_resources'])
    write_json(BUNDLE/'preservation_audit.json',closure['historical_preservation'])
    modules=__import__('current_upstream').authenticate_gate2()
    write_json(BUNDLE/'segmentation_compatibility_audit.json',segmentation_compatibility_audit(modules))
    write_json(BUNDLE/'preparation_initialize_success.json',dict(status='ALL_PRE_SOURCE_PRODUCTION_GATES_PASS',
        full_archive_hash_invoked=False,pending_created=False,guard_counters=counters,activity=dict(ZERO)))
    print('V3_INITIALIZED_ALL101_CLEAN_AND_INVENTORY_PASS; NO_FULL_SOURCE_HASH_YET',flush=True)


def seal(attempt):
    counters=install_guard('prepare',source_integrity=True)
    future_namespaces_absent();require(not (BUNDLE/'SHA256SUMS').exists(),'V3 already sealed')
    tests=read_json(BUNDLE/'synthetic_test_results.json')
    require(tests['attempt']==attempt and tests['passed']==tests['tests_run'] and
        tests['failed']==tests['errors']==tests['skipped']==0,'All accepted tests must PASS')
    python={p.name:hash_file(p) for p in sorted(BUNDLE.glob('*.py'))}
    require(tests['bundle_python_sha256']==python,'Implementation changed after tests')
    integration=read_json(BUNDLE/'production_integration_test_results.json')
    require(integration['status']=='PASS' and integration['failed']==integration['errors']==integration['skipped']==0,
        'Production integration tests failed')
    order=read_json(BUNDLE/'execution_order_audit.json')
    require(order['status']=='PASS' and len(order['six_predictable_failure_injections'])==6 and
        all(r['full_source_hash_calls']==r['pending_create_calls']==0 for r in order['six_predictable_failure_injections']),
        'Cheap failure ordering not proven')
    verify_amendment()
    closure=pre_source_closure(None,preparing=True,progress=lambda message:print(message,flush=True))
    require(closure['production_clean_resolution']==read_json(BUNDLE/'production_clean_resolution_audit.json') and
        closure['production_inventory_schema']==read_json(BUNDLE/'production_inventory_schema_audit.json'),
        'Production clean/inventory evidence changed since tests')
    write_json(BUNDLE/'pre_seal_execution_path_closure_audit.json',closure)
    source=complete_source_integrity(closure,lambda message:print(message,flush=True))
    write_json(BUNDLE/'source_integrity_audit.json',source)
    future_namespaces_absent()
    require(git_state()==read_json(BUNDLE/'preparation_initial.json')['git_state'],'Git changed')
    changes=write_diff()
    old={p.relative_to(V2_BUNDLE).as_posix() for p in V2_BUNDLE.rglob('*') if p.is_file()}
    predicted={'source_code_audit.json','preparation_audit.json','bundle_inventory.json','SHA256SUMS','SHA256SUMS.sha256'}
    current={p.relative_to(BUNDLE).as_posix() for p in BUNDLE.rglob('*') if p.is_file()}|predicted
    exact=[name for name in sorted(old&current) if (BUNDLE/name).is_file() and hash_file(BUNDLE/name)==hash_file(V2_BUNDLE/name)]
    baseline=read_json(BUNDLE/'preservation_baseline.json')
    exporter=next(r for r in baseline['files'] if r['path']==HISTORICAL_EXPORTER.relative_to(REPO).as_posix())
    write_json(BUNDLE/'source_code_audit.json',dict(status='PASS',new_bundle_python_sha256=python,
        changes_relative_to_v2=changes,all_file_changes_relative_to_v2=dict(
            modified_v3_counterparts=sorted((old&current)-set(exact)),new_v3_files=sorted(current-old),
            byte_identical_v2_copies=exact,old_v2_evidence_not_relabelled=sorted(old-current),
            v2_files_deleted_or_modified=False),
        exact_modified_new_python_sha256={name:python[name] for name in
            changes['modified_python_relative_to_v2']+changes['new_python_relative_to_v2']},
        scientific_files_changed=[],scientific_invariance='PASS',current_upstream_and_restoration_byte_identical=True,
        graph_replay_export_model_training_byte_identical=True,subprocess_policy_byte_identical=True,
        historical_N20_exporter_unchanged=True,historical_N20_exporter=exporter,
        all_previous_outputs_preserved=True,v1_pending_inventory_sha256=verify_v1_failure()['inventory_sha256'],
        exact_v2_preserved_files=inventory(V2_BUNDLE),v1_seal=V1_BUNDLE_SEAL,v2_seal=V2_BUNDLE_SEAL))
    write_json(BUNDLE/'preparation_audit.json',dict(status='V3_PREPARATION_ONLY_COMPLETE_STOP_FOR_HUMAN_REVIEW',
        scientific_invariance='PASS',tests={k:tests[k] for k in ('passed','failed','errors','skipped','tests_run')},
        production_integration_tests={k:integration[k] for k in ('passed','failed','errors','skipped')},
        real_context_status='PASS',production_clean_counts=integration['clean_counts'],production_inventory_records_checked=101,
        canonical_summaries_checked=101,canonical_digests_checked=101,fake_record_sha256_dependency=False,
        segmentation_compatibility='PASS',disk_resource='PASS',execution_order='PASS',
        full_source_integrity=source,historical_preservation=closure['historical_preservation'],
        v1_preserved=True,v2_preserved=True,v1_pending_preserved=True,v1_reproduction_preserved=True,
        v1_pending=verify_v1_failure(),all_real_models_instantiated=False,
        data_v3_exists=False,pending_v3_exists=False,runs_v3_exists=False,
        graph_data_generation_executed=False,real_model_training_executed=False,activity=dict(ZERO),
        git_state=closure['git_state'],changes_relative_to_v2=changes,guard_counters=counters))
    require(all((BUNDLE/name).is_file() for name in REQUIRED if name not in
        ('bundle_inventory.json','SHA256SUMS','SHA256SUMS.sha256')),'Missing required named evidence')
    value=seal_tree(BUNDLE)
    for path in BUNDLE.rglob('*'):
        if path.is_file():path.chmod(stat.S_IREAD)
    require(all(p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY for p in BUNDLE.rglob('*') if p.is_file()),'V3 not read-only')
    print(json.dumps(dict(status='V3_SEALED_STOP_FOR_HUMAN_REVIEW',bundle_path=str(BUNDLE),bundle_seal=value,
        tests={k:tests[k] for k in ('passed','failed','errors','skipped')},production_integration_tests=integration['passed'],
        clean_counts=integration['clean_counts'],scientific_invariance='PASS',activity=dict(ZERO))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['initialize','seal']);parser.add_argument('--test-attempt',type=int)
    args=parser.parse_args()
    if args.action=='initialize':initial()
    else:require(args.test_attempt is not None,'Accepted test attempt required');seal(args.test_attempt)
