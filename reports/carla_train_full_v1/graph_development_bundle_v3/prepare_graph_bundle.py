"""Preparation lifecycle only: inspect -> synthetic tests in separate process -> seal -> human review."""
import argparse
import difflib
import datetime
from graph_common import *
from current_upstream import verify_current,same_current,protocol_identity
from current_restore import inspect_current_states

LIMITATION='Current graph validation is development-only: upstream one-class representation was fitted using all 76 FIT scenarios, including the 15 GRAPH_VAL scenarios. Validation is reused for early stopping and threshold selection; constructed pseudo distribution and artificial paired 1:1 prevalence do not establish held-out or physical-safety performance.'


def leaves(value,pointer=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from leaves(v,pointer+'/'+k.replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from leaves(v,pointer+'/'+str(i))
    else:yield dict(json_pointer=pointer,value=value)


def documents(bindings,restore):
    identity,protocol=protocol_identity()
    bindings_excluded=[dict(record,classification='NOT CURRENT',reason='Historical registration identity only; current independently verified binding is in current_experiment_bindings.json')
                       for record in leaves(protocol['bindings'],'/bindings')]
    changes={
        '/FIT_graph_pseudo_policy/future_adapter_contract':'Replaced N20 cached adapter with current v8/v7 canonical clean resolver, current Gate-2 v3 states/mapping and authenticated full TRAIN replay',
        '/FIT_graph_pseudo_policy/balance':'Old N20 maxima NOT CURRENT; current upper bounds: GRAPH_TRAIN 182939 pairs /365878 rows, GRAPH_VAL 44985 pairs /89970 rows, before no-effect skips',
        '/training_specification/validation_loss':'Equal-scenario mean BCE over CURRENT frozen 15 GRAPH_VAL; historical 3-scenario wording NOT CURRENT',
        '/conformal_placement/CAL_scenarios_reserved_from_graph_development':'Historical five-CAL membership NOT CURRENT; current 25 CAL excluded; conformal entirely deferred',
        '/conformal_placement/limitation':'Historical five-CAL wording NOT CURRENT; all current 25 CAL were used by upstream mappings, so naive reuse supplies no independent guarantee',
        '/interpretation_limits/3':'Historical three-validation-scenario limitation NOT CURRENT; use current 15 GRAPH_VAL development limitation',
        '/GNSS_secondary':'Deferred and not part of this primary three-node preparation; no GNSS data/features/mapping/models',
        '/authorization_boundary':'Current human request authorizes preparation and synthetic fixtures only; never real data or training',
        '/readiness':'Historical readiness is NOT CURRENT execution authorization; current data/runs must remain absent',
        '/frozen_at_utc':'Historical registration timestamp only, not current preparation time'}
    dispositions=[]
    for key in protocol:
        dispositions.append(dict(json_pointer='/'+key,classification='HISTORICAL_BINDING_NOT_CURRENT' if key=='bindings' else
            'ADAPTED_CURRENT_SCOPE' if any(p.startswith('/'+key+'/') or p=='/'+key for p in changes) else
            'FROZEN_SCIENTIFIC_DEFINITION_ONLY',execution_authority=False))
    adaptation=dict(protocol_identity=identity,all_historical_binding_leaves=bindings_excluded,
        all_top_level_sections=dispositions,explicit_adaptations=changes,old_historical_graph_partition='NOT CURRENT',
        old_execution_outputs='NOT CURRENT; protected evidence only; no scores, epochs, thresholds, outcomes or rankings parsed',
        preserved_science_sections=['GAT_architecture','NoGraph','base_node_features','epistemic_attention',
            'node_order','primary_methods','information_matching','applicable training_specification fields',
            'applicable FIT_graph_pseudo_policy fields'],current_binding_source='current_experiment_bindings.json',
        no_old_N20_paths_opened=True,no_historical_performance_used=True)
    write_json(BUNDLE/'historical_protocol_adaptation.json',adaptation)
    write_json(BUNDLE/'historical_evidence_exclusion.json',dict(status='PROTECTED HISTORICAL EVIDENCE ONLY',
        protected_namespaces=['reports/carla_gat_paired_execution_v1/','reports/carla_gat_paired_raw_runs_v1/',
            'all pre-existing graph runs, final-evaluation and conformal report namespaces'],
        preservation_scope='every pre-existing report namespace; exact opaque hashes and write metadata only',
        old_N20_bindings='ALL NOT CURRENT, enumerated leaf-by-leaf in historical_protocol_adaptation.json',
        historical_N20_exporter=str(HISTORICAL_EXPORTER),historical_exporter_run=False,historical_exporter_modified=False,
        rankings_scores_epochs_thresholds_outcomes_used=False,official_TEST_requested=False))
    science=dict(schema='Experiment-2B-graph-science-v1',primary_methods=METHODS,paired_seeds=SEEDS,
        node_order=NODE_ORDER,node_feature_order=FEATURE_ORDER,source_computation_dtype='float64',model_dtype='float32',
        fitted_feature_scaler=None,adjacency=ADJACENCY_LIST,directed_edges=6,self_loops=0,
        NoGraph=protocol['NoGraph'],GAT_architecture=protocol['GAT_architecture'],
        epistemic_attention=protocol['epistemic_attention'],original_sender_E_at_both_layers=True,
        Standard_Epistemic_identical_cloned_initialization=True,
        interpretation='P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo task; never physical safety probability',
        validation_limitation=LIMITATION,minimum_AUROC_F1_accuracy=None,Epistemic_required_to_beat_Standard=False,
        negative_and_null_results_visible=True,scientific_repair_after_observed_current_results='Separately numbered experiment; no invisible rescue')
    write_json(BUNDLE/'graph_science_spec.json',science)
    export=dict(schema='Experiment-2B-graph-export-v1',parents='CURRENT frozen FIT_NORMAL only',
        FIT_count=76,GRAPH_TRAIN_count=61,GRAPH_VAL_count=15,current_membership='current_partition_audit.json',
        current_partition_sha256=GRAPH_PARTITION_SHA,CAL_pseudo_reuse=False,CAL_parent_count=0,role_inheritance='exact parent graph role',
        ticks=dict(first=1,last=2999,one_candidate_per_tick=True),recipe_order=RECIPES,selection='recipe_order[t % 5]',
        severities=SEVERITIES,generator_base_seed=0,per_tick_seed='t',fallback=False,resampling=False,
        no_effect=dict(equation='np.allclose(corrupted_compact,parent_compact,rtol=1e-12,atol=1e-12)',
            action='DROP_BOTH before p/E/A scoring',reason='compact_modality_no_effect_np_allclose_rtol_1e-12_atol_1e-12'),
        mapping_collisions='RETAIN_PAIR and log canonical_pEA_collision; never select by score or p/E/A',
        targets=dict(clean=1,pseudo=0),corrupted_compact_nodes=1,untouched_pEA='byte-identical',
        IMU_window='[max(0,t-11),t]',Camera_Seg_window='instantaneous t',clean_features_recomputed=False,
        clean_feature_store='Canonical authenticated native v8 (33) / identity-adopted v7 (68) blocks',
        clean_pEA='Generated through current Gate-2 v3 restored members/mapping; no N20 state',
        segmentation_sanitize='unchanged current v8 sanitize before recipe',upstream_refit=False,recalibration=False,
        raw_replay='Unchanged authenticated TrainReplay, full compressed SHA/bytes, gzip EOF/TAR end, 101 exact scenario index records and every member hash/chain; FIT-only bounded scratch',
        raw_member_cap_bytes=128<<20,free_disk_reserve_bytes=20<<30,max_raw_scenarios_materialized=1,
        scratch_ownership='new pending namespace only; failed pending retained, never resume/retry silently',
        future_data_namespace=str(DATA),future_run_namespace=str(RUNS),real_export_authorized=False,
        row_order='archive scenario ordinal, then ascending tick, then clean/pseudo',
        pair_id='SHA256 canonical schema/scenario/tick + current partition SHA + Gate2 runtime seal + scientific protocol SHA',
        row_key='pair_id + :clean or :pseudo',scientific_content_hash='Canonical metadata/ledger plus ordered float64 little-endian source node bytes; independent of NPZ container compression/seal',
        per_scenario_artifacts=['graphs.npz','rows.json','candidate_ledger.json','summary.json','artifact_inventory.json','SHA256SUMS','SHA256SUMS.sha256'],
        publication='Only after complete 76 FIT coverage and successful full-source replay; exclusive pending then non-overwriting rename',
        maximum_pairs_before_skips=dict(GRAPH_TRAIN=61*2999,GRAPH_VAL=15*2999),interpretation=science['interpretation'])
    write_json(BUNDLE/'graph_export_spec.json',export)
    training=dict(schema='Experiment-2B-graph-training-v1',batch_size_graphs=256,optimizer='Adam',learning_rate=.001,
        weight_decay=.0001,maximum_epochs=100,early_stopping=dict(patience=10,min_delta=1e-5,minimum_epochs=1,
        tracker_reset='loss < tracked_best - min_delta',checkpoint='actual lowest validation BCE among every executed epoch; earliest exact tie',
        separate_checkpoint_and_tracker=True),loss='mean BCE(q_normal,target_normal) over GRAPH_TRAIN samples; no node-label auxiliary loss',
        validation_loss='equal-scenario mean BCE over CURRENT frozen 15 GRAPH_VAL',current_GRAPH_VAL=bindings['partition']['GRAPH_VAL'],
        training_precision='float32',development_statistics_precision='float64 NumPy BCE with separate q and (1-q) clipping [1e-7,1-1e-7], matching applicable historical metric definition',
        AMP=False,TF32=False,gradient_clipping=None,scheduler=None,shuffle='Generator(PCG64(seed + zero_based_epoch_index)).permutation',
        same_row_keys_and_epoch_order_for_all_methods=True,seeds=SEEDS,best_seed_selection=False,seed_replacement=False,
        upstream_agents_frozen=True,real_execution_authorized=False,synthetic_smoke=dict(maximum_rows=512,maximum_epochs=2,device='CPU',
        scope='Fixture-only verification; does not alter the frozen future maximum_epochs=100'),future_device='single T4; no distributed training; separately lock/validate environment',
        validation_limitation=LIMITATION)
    write_json(BUNDLE/'training_spec.json',training)
    write_json(BUNDLE/'threshold_spec.json',dict(checkpoint='selected actual minimum-BCE checkpoint',separate_for='each method/seed',
        candidates='sorted unique CURRENT GRAPH_VAL p_corruption union {0,1}',p_corruption='1-q_normal',
        corruption_iff='p_corruption >= tau',objective='arithmetic mean of per-current-15-GRAPH_VAL-scenario corruption F1',
        exact_tie='largest tau',F1='2TP/(2TP+FP+FN); zero denominator -> 0',development_only=True,
        architecture_recipes_membership_seeds_training_spec_may_change_based_on_threshold_performance=False,
        current_GRAPH_VAL=bindings['partition']['GRAPH_VAL'],minimum_required_F1=None))
    write_json(BUNDLE/'graph_training_authorization_design.json',dict(status='DEFERRED; HUMAN REVIEW REQUIRED',
        model_execution_authorized=False,real_training_entrypoint_available=False,
        next_step='Separate data-only authorization using sealed generate_graph_data.py; review immutable data artifact before any model authorization',
        required_future_authorization=['explicit human authorization for graph model execution',
            'independent current preparation bundle external seal','independent data seal and scientific content hash',
            'read-only data validation, exact 61/15 role/paired rows/current state/scientific source checks',
            'fresh environment lock and deterministic single-T4 fixture checks matching frozen model/training/threshold specification',
            'a separately reviewed execution adapter; this bundle exposes synthetic model/trainer startup only'],
        future_namespace=str(RUNS),run_plan=[dict(seed=s,method=m) for s in SEEDS for m in METHODS],
        observed_outputs='All 15 paired method/seed outcomes, epoch losses, actual selected checkpoint, development thresholds and null/negative results retained',
        failure_policy='Fail closed; no replacement seed/scenario or silent rerun; retain failure evidence',
        minimum_AUROC_F1_accuracy=None,Epistemic_required_to_win=False,scientific_repair='Separately numbered experiment after observed results',
        TEST_or_final_evaluation_or_conformal_authorized=False,validation_limitation=LIMITATION))
    python=bindings['environment']['python_executable']
    commands=dict(read_only_preflight=f'& "{python}" -B "{BUNDLE / "preflight_graph.py"}" --bundle-seal <reviewed-bundle-seal>',
        future_graph_data_generation=f'& "{python}" -B "{BUNDLE / "generate_graph_data.py"}" --bundle-seal <reviewed-bundle-seal> --authorize-graph-data-generation {DATA_TOKEN}',
        graph_data_command_executed=False,graph_training_command_executed=False,authorization_granted=False,
        external_seal='Use exact external SHA256SUMS seal from the preparation completion message, not a recalculated or changed bundle')
    write_json(BUNDLE/'future_commands.json',commands)
    readme=f'''# Experiment-2B FULL-TRAIN graph-development preparation

Preparation only. STOP FOR HUMAN REVIEW. No real FIT pseudo, graph artifact or model initialization/training is authorized or performed. Data and run namespaces must be absent.

Current bindings are independently authenticated in current_experiment_bindings.json. Frozen graph partition SHA: {GRAPH_PARTITION_SHA}; namespace seal: {SEALS['graph_partition_freeze_v1']}. Membership is read/asserted without RNG recomputation. Historical protocol SHA {PROTOCOL_SHA} authenticates exact committed LF bytes; the working CRLF file is separately hashed and proven EOL-equivalent without modification.

Historical N20 bindings are all NOT CURRENT and are listed individually in historical_protocol_adaptation.json. Historical outputs are protected prior execution evidence; preservation hashes are opaque and no performance/ranking inputs are used. The historical exporter remains unchanged and is never imported or run.

The new adapter resolves existing clean features with the authenticated Gate-1 v8/v7 resolver. Future-only restoration constructs the exact five saved Mahalanobis members per modality without constructors/fit and restores the accepted EnsemblePredictiveCalibrator a,b,mean,std plus fitted/validity metadata. The raw-score mapping is sigmoid(a*s+b); mean/std must not be applied again. current_gate2_restore_audit.json records exact state shapes/hashes and JSON/NPZ equality. Synthetic restoration fixtures verify the production score and uncertainty equations.

graph_export.py implements one candidate per FIT tick, fixed recipes/severities, no-effect pair removal, collision retention, exact role inheritance and untouched-node byte identity. graph_replay.py uses unchanged authenticated TrainReplay and inventory functions, captures at most one scenario's scheduled raw FIT inputs, and publishes only after complete archive/member/chain/101-scenario verification. No clean feature recomputation or full raw archive copy occurs. Future failures retain pending evidence and require human review.

graph_models.py and graph_training.py are guarded synthetic-only implementations. They exercise exact generic attention definitions, 48/50/50 parameters, five paired seeds, independent checkpoint/tracker rules and development threshold mechanics. They cannot start a real run. Training authorization and a real execution adapter are deferred as described in graph_training_authorization_design.json.

{LIMITATION}

There is no minimum AUROC/F1/accuracy, no required EpistemicGAT win, no best-seed selection/replacement, and negative/null results remain visible. Any scientific repair after observing current graph results requires a separately numbered experiment.

SHA256SUMS, detached SHA256SUMS.sha256 and bundle_inventory.json authenticate every bundle file and directory. Files are set read-only after sealing. Read-only preflight checks external seal, current state/partition/source hashes, exact historical preservation, tested source hashes, absent future namespaces and read-only attributes. It fresh-hashes the complete TRAIN archive for integrity only; it never parses/replays payload. Preparation sealing reuses the independently measured initial full hash only while source identity metadata remain identical.

Future commands (NOT authorization; replace the seal placeholder with the externally reviewed seal):

```powershell
{commands['read_only_preflight']}
{commands['future_graph_data_generation']}
```

The second command must be separately authorized by the human. It grants data generation only, never graph training. No real training command is provided in this preparation.
'''
    write_new(BUNDLE/'README.md',readme.encode())


def initialize():
    future_namespaces_absent();require(not (BUNDLE/'preparation_initial.json').exists(),'Preparation initialization exists')
    counters=install_guard('prepare',source_integrity=True)
    initial=git_state();require(initial['HEAD']==HEAD and not initial['tracked'] and not initial['staged'],'Git dirty/HEAD mismatch')
    if (BUNDLE/'preservation_baseline.json').exists():
        require((BUNDLE/'preparation_failure_1.json').exists(),'Existing baseline without retained failure record')
        baseline=read_json(BUNDLE/'preservation_baseline.json')
        print('Reusing original opaque preservation baseline from retained verifier-only failure; full comparison at sealing',flush=True)
    else:
        baseline=preservation_snapshot(lambda message:print(message,flush=True))
        write_json(BUNDLE/'preservation_baseline.json',baseline)
    bindings=verify_current(full_source_hash=True,progress=lambda message:print(message,flush=True))
    restore=inspect_current_states()
    write_json(BUNDLE/'current_experiment_bindings.json',bindings)
    write_json(BUNDLE/'current_partition_audit.json',bindings['partition'])
    write_json(BUNDLE/'current_gate2_restore_audit.json',restore)
    documents(bindings,restore)
    write_json(BUNDLE/'preparation_initial.json',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_state=initial,source_full_sha256=bindings['source_full_sha_independently_measured'],guard_counters=counters,
        preservation_baseline_sha256=hash_file(BUNDLE/'preservation_baseline.json'),activity=dict(ZERO)))
    print('INITIAL_PREPARATION_COMPLETE; synthetic tests may now run; real data/runs remain absent',flush=True)


def seal(attempt):
    counters=install_guard('prepare',source_integrity=True)
    future_namespaces_absent();require(not (BUNDLE/'SHA256SUMS').exists(),'Already sealed; do not change')
    bindings=read_json(BUNDLE/'current_experiment_bindings.json')
    tests=read_json(BUNDLE/f'synthetic_test_attempt_{attempt}.json')
    require(tests['failed']==tests['errors']==tests['skipped']==0 and tests['passed']==tests['tests_run'],'Required synthetic tests did not pass')
    python={p.name:hash_file(p) for p in sorted(BUNDLE.glob('*.py'))}
    require(tests['bundle_python_sha256']==python,'Python source changed after synthetic tests; rerun needed')
    current=verify_current(progress=lambda message:print(message,flush=True));same_current(current,bindings)
    require(inspect_current_states()==read_json(BUNDLE/'current_gate2_restore_audit.json'),'State audit changed')
    preservation=verify_preservation(read_json(BUNDLE/'preservation_baseline.json'),lambda message:print(message,flush=True))
    before=read_json(BUNDLE/'preparation_initial.json')['git_state'];after=git_state()
    require(after==before,'HEAD/status/tracked/staged changed unexpectedly')
    baseline=read_json(BUNDLE/'preservation_baseline.json')
    historical_record=next(r for r in baseline['files'] if r['path']==HISTORICAL_EXPORTER.relative_to(REPO).as_posix())
    write_json(BUNDLE/'source_code_audit.json',dict(status='PASS',new_bundle_python_sha256=python,
        current_scientific_dependency_sha256=current['scientific_source_hashes'],
        historical_exporter_before_after=historical_record,historical_exporter_unchanged=True,
        historical_exporter_imported_or_executed=False,tracked_sources_added_or_changed=[],
        exact_new_python_files=[str(BUNDLE/name) for name in python],source_dependency_scope='Exact working-byte hashes for current scientific core types/interface, normality/agents/features/recipes/split/generic layer and every current Gate1 v7/v8 and Gate2 v3 report-side Python dependency',
        authentication='Pinned upstream seals before import; exact module origins; exact unedited scientific AST bodies; current working hashes independently bound',
        old_N20_execution_dependencies_imported=False,real_model_startup_forbidden=True))
    diff=[]
    for path in sorted(BUNDLE.glob('*.py')):
        rel=path.relative_to(REPO).as_posix()
        diff.extend(difflib.unified_diff([],path.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
    write_new(BUNDLE/'engineering.diff',''.join(diff).encode())
    write_json(BUNDLE/'preparation_audit.json',dict(status='PREPARATION_ONLY_COMPLETE_STOP_FOR_HUMAN_REVIEW',
        accepted_test_evidence=f'synthetic_test_attempt_{attempt}.json',adversarial_evidence=f'adversarial_test_attempt_{attempt}.json',
        tests={k:tests[k] for k in ('passed','failed','errors','skipped','tests_run')},
        all_attempts_retained=sorted(p.name for p in BUNDLE.glob('*test_attempt_*.json')),
        current_upstream_hashes_verified=True,source_full_SHA_independently_verified=True,
        sealing_source_check='Initial independent full SHA receipt plus identical fingerprint; fresh full hash on future preflight',
        historical_protocol_exact_committed_SHA_verified=True,working_file_CRLF_equivalence_verified=True,
        membership_recomputed=False,historical_preservation=preservation,historical_N20_exporter_unchanged=True,
        historical_graph_outputs_untouched=True,historical_performance_used=False,current_Gate2_restoration=read_json(BUNDLE/'current_gate2_restore_audit.json'),
        graph_partition_sha256=GRAPH_PARTITION_SHA,graph_partition_seal=SEALS['graph_partition_freeze_v1'],
        graph_data_runtime_exists=False,graph_run_runtime_exists=False,activity=dict(ZERO),git_state=after,
        guard_counters=counters,graph_data_generation_authorized=False,graph_training_authorized=False,
        read_only_preflight_available=True,validation_limitation=LIMITATION,immutable_after_sealing='Every file set FILE_ATTRIBUTE_READONLY; exact external SHA256 seal detects changes/additions/removals',
        next_action='STOP FOR HUMAN REVIEW'))
    bundle_seal=seal_tree(BUNDLE)
    for path in BUNDLE.rglob('*'):
        if path.is_file():path.chmod(stat.S_IREAD)
    require(all(p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY for p in BUNDLE.rglob('*') if p.is_file()),'Read-only setting failed')
    print(json.dumps(dict(status='SEALED_PREPARATION_STOP_FOR_HUMAN_REVIEW',bundle_path=str(BUNDLE),bundle_seal=bundle_seal,
        python_files=sorted(python),tests={k:tests[k] for k in ('passed','failed','errors','skipped')},activity=dict(ZERO))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['initialize','seal']);parser.add_argument('--test-attempt',type=int)
    args=parser.parse_args()
    if args.action=='initialize':initialize()
    else:
        require(args.test_attempt is not None,'Accepted synthetic attempt required');seal(args.test_attempt)
