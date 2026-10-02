"""Seal a bounded, incomplete metadata audit. No allocation, score or fit."""
import json
import math
from pathlib import Path
from audit_metadata import HERE, ROOT, PRE, EXPECTED, sha, read, write, git, now

TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD']
EXCLUSIONS = ['test/anomaly/Town01/change-weather/scenario-1', 'test/anomaly/Town01/change-weather/scenario-10']
SOURCE = 'https://carlanomaly.de/dataset/'
BLOCKER = 'No standalone official scenario manifest or route/seed/family catalogue is exposed by the inspected publisher indexes. Complete archive traversal would decompress intervening TEST payloads and is prohibited. Stop official access.'

def main():
    assert read(HERE / 'preregistration_verification.json')['passed']
    assert sha(PRE / 'SHA256SUMS') == EXPECTED
    assert read(PRE / 'test_partition_protocol.json')['exact_excluded_scenario_ids'] == EXCLUSIONS
    doc = read(HERE / 'composition_and_directory_documentation.json')
    intro = doc['sections']['introduction_composition']
    assert 'Train101303,00060' in intro and 'Test normal10732,10060' in intro and 'Test anomaly520156,00069' in intro
    assert 'Town01' in doc['sections']['directory_structure'] and 'Town10HD' in doc['sections']['directory_structure']
    common = {'milestone': 'OFFICIAL TEST METADATA INVENTORY / PROVENANCE / COUNT AUDIT ONLY',
              'protocol_manifest_sha256': EXPECTED, 'time_utc': now(), 'client_date': '2026-10-02', 'client_timezone': 'Asia/Calcutta'}
    exclusions = []
    for sid, extent in zip(EXCLUSIONS, ['complete', 'partial']):
        exclusions.append({'scenario_id': sid, 'town': 'Town01', 'official_directory_condition': 'anomaly',
            'anomaly_type': 'change-weather', 'provenance_family_group_id': None,
            'metadata_identity_source': 'Sealed amended historical exposure record; not independently enumerated in this audit',
            'historical_exposure_extent': extent, 'classification': 'SCHEMA_EXPOSED_DIAGNOSTIC_ONLY',
            'eligible': False, 'exclusion_reason': 'Frozen historical schema exposure; entire scenario permanently excluded from both final sets',
            'excluded_roles': ['FINAL_CONFORMAL_CAL', 'FINAL_EVALUATION'],
            'official_existence_independently_verified_this_milestone': False})
    write('test_scenario_inventory.json', {**common, 'status': 'INCOMPLETE_METADATA_UNAVAILABLE',
        'complete': False, 'scenario_ids_enumerated_from_official_index': [], 'discovered_scenario_count': None,
        'known_excluded_scenario_records': exclusions, 'additional_exposed_ids_inferred': [],
        'scenario_directory_completion_verified': False, 'blocker': BLOCKER,
        'directory_grammar': ['test/normal/<town>/scenario-N', 'test/anomaly/<town>/<anomaly-type>/scenario-N']})
    write('historical_exclusion_verification.json', {**common, 'records': exclusions,
        'identity_matches_amended_record': True, 'official_path_existence_verification': 'NOT_VERIFIED_FROM_INDEPENDENT_OFFICIAL_SCENARIO_METADATA',
        'historical_file_level_ledger_complete': False, 'additional_exposed_ids_established': [],
        'sensor_and_timestep_label_contents_read': False, 'tuning_use': False})
    write('test_counts.json', {**common, 'status': 'PUBLISHER_AGGREGATES_ONLY_NOT_ENUMERATED',
        'source': SOURCE, 'total_discovered_TEST_scenarios': None,
        'publisher_reported': {'TEST_total': 627, 'normal': 107, 'anomaly': 520, 'normal_towns': 6, 'anomaly_towns': 6, 'anomaly_types': 9},
        'eligible_after_known_exclusions_conditional_on_publisher_inventory_and_two_frozen_identities': {'total': 625, 'normal': 107, 'anomaly': 518},
        'eligibility_qualification': 'Arithmetic removal of the two frozen historical IDs only; not a completed inventory/exposure clearance.',
        'towns_documented': TOWNS,
        'by_town': [{'town': t, 'normal': None, 'anomaly': None, 'total': None, 'eligible': None} for t in TOWNS],
        'by_anomaly_type': None, 'by_town_condition_anomaly_type': None,
        'missing_counts': ['town composition', 'per-type counts', 'town x condition x type counts', 'verified eligible counts'],
        'no_timestep_counts_or_labels_used': True})
    provenance = [
        {'relationship': 'generation platform', 'classification': 'VERIFIED', 'evidence': 'Publisher introductory metadata states CARLA generation', 'scope': 'Dataset-level, not independent seed/route identity'},
        {'relationship': 'split/town/type directory hierarchy', 'classification': 'VERIFIED', 'evidence': 'Publisher directory-structure documentation', 'scope': 'Naming grammar only; schematic scenario-1 is not an exhaustive listing'},
        *[{'relationship': r, 'classification': 'UNKNOWN', 'evidence': 'No explicit scenario-level keys or mapping available in inspected metadata'} for r in ['same route', 'same simulator seed', 'normal/anomaly versions of same drive', 'weather variants sharing a base drive', 'anomaly variants sharing a base drive', 'shared generation family', 'explicit per-scenario provenance key']],
        {'relationship': 'generation configuration could be available through CarlaGen', 'classification': 'SUPPORTED BUT INCOMPLETE',
         'evidence': 'Publisher resource catalogue describes Hydra configs and CARLA API; its CarlaGen link is # and supplies no configuration/family mapping'}]
    write('provenance_audit.json', {**common, 'sources': [SOURCE, 'https://carlanomaly.de/resources/', 'https://carlanomaly.de/about/', 'https://data.carlanomaly.de/v1/'],
        'relationships': provenance, 'reliable_grouping_identifier_available': False, 'no_sensor_similarity_inference': True,
        'release_binding': {'archive_checksum_catalogue': 'official_archive_checksums.json', 'checksum_verified_against_archive_bytes': False},
        'publisher_does_not_supply_seed_route_independence_evidence_in_inspected_pages': True})
    proposal = [
        'Obtain and seal a publisher-issued scenario-only catalogue with explicit base-drive/family/route/seed keys, without sensor or timestep-label fields.',
        'Build an undirected graph on full split-qualified IDs using only verified explicit linkage keys; connected components are indivisible groups. Do not group merely equal scenario-N suffixes or common towns/types.',
        'Exclude the two historically exposed scenarios. If verified links make entire-family exclusion necessary, specify it in a separately sealed exposure amendment; do not silently infer additional exposed IDs.',
        'For families crossing town/type/condition strata, define the family sampling target and a deterministic whole-family allocation objective. Exact ceil(N_h/5) scenario quotas may be incompatible; any relaxation or new grouping strata must be preregistered.',
        'After an authorized amendment, fix sorted components, RNG/version and one group assignment shared by all methods/seeds; prevent any linked component crossing roles. Use family-level calibration units where dependence requires it and re-audit rank/count adequacy.',
        'Seal the proposed algorithm before predictions or payload access; execute only with separate authorization.'
    ]
    write('family_grouping_audit.json', {**common, 'classification': 'UNKNOWN', 'verified_groups': [],
        'group_ids_manufactured': False, 'reliable_grouping_identifier_available': False,
        'normal_anomaly_latent_families': 'UNKNOWN', 'route_seed_duplication_crossing_CAL_EVAL': 'CANNOT_BE_RULED_OUT',
        'town_type_stratification_defensible_for_new_scenario_exchangeability': 'NOT_ESTABLISHED',
        'grouping_before_partition_if_links_found': True, 'conditional_group_level_split_proposal_not_executed': proposal,
        'between_scenario_exchangeability': 'NOT_VERIFIED; town/type composition and deterministic split seed do not establish independence/exchangeability'})
    normal_rows = [{'town': t, 'condition': 'normal', 'anomaly_type': 'NORMAL', 'N_h': None,
                    'CAL_count': None, 'EVAL_count': None, 'cutoff_is_infinite': None} for t in TOWNS]
    write('stratum_count_audit.json', {**common, 'status': 'EXACT_STRATUM_COUNTS_UNAVAILABLE',
        'registered_stratum': '(town, official directory condition, anomaly-type or NORMAL)',
        'normal_strata': normal_rows, 'anomaly_strata': None,
        'quota': 'n_cal=(N_h+4)//5; n_eval=N_h-n_cal', 'rank': 'k=(19*(n_cal+1)+19)//20',
        'minimum_CAL_for_noninfinite_augmented_rank': 19, 'minimum_eligible_N_h': 91,
        'exact_sparse_strata': None, 'normal_sparse_strata_lower_bound': 5,
        'reason': 'Six nonempty normal town strata sum to 107; at most one can have N_h>=91. At least five have N_h<=90 and n_cal<19. At least one has N_h<=floor(107/6)=17 and n_cal<=4.'})
    # Count algebra only. No IDs are shuffled, assigned or loaded into an RNG.
    # For six positive integer town counts totalling 107, ceil quotas sum between
    # ceil(107/5)=22 and floor((107+4*6)/5)=26. Both bounds are attainable.
    assert math.ceil(107 / 5) == 22 and (107 + 24) // 5 == 26
    assert sum((n + 4) // 5 for n in [82, 5, 5, 5, 5, 5]) == 22
    assert sum((n + 4) // 5 for n in [81, 6, 6, 6, 6, 2]) == 26
    partition = {**common, 'status': 'COUNT_BOUNDS_ONLY_EXACT_SIMULATION_BLOCKED', 'actual_partition_executed': False,
        'randomized_memberships_published': False, 'RNG_created': False, 'partition_seed_unchanged': 2028,
        'normal_CAL_total_bounds': [22, 26], 'normal_EVAL_total_bounds': [81, 85],
        'total_CAL_and_EVAL_counts': None, 'per_anomaly_stratum_counts': None,
        'bound_examples': 'Integer vectors used solely to validate attainable arithmetic bounds; no vector is asserted to be actual town composition.',
        'singleton_rule': 'N_h=1 would allocate one CAL and zero EVAL; occurrence unknown',
        'complete_scenario_identity_gate_passed': False, 'historical_exposure_reconciliation_gate_passed': False,
        'family_exchangeability_gate_passed': False, 'protocol_modified': False}
    write('partition_feasibility.json', partition)
    feasibility = {**common, 'status': 'GLOBAL_INFINITY_NECESSARY_CONDITIONAL_ON_PUBLISHER_AGGREGATES',
        'is_performance_evaluation': False, 'alpha_unchanged': 0.05,
        'scope': 'Mathematical implication of published normal count and six nonempty town strata, not a fitted result or verified inventory.',
        'Q_global_necessarily_infinite_under_published_counts': True,
        'applies_to_every_frozen_method_and_seed': True, 'method_seed_pairs': 15,
        'normal_sparse_strata_at_least': 5, 'exact_culprit_strata_and_counts': None,
        'candidate_culprit_strata': [f'({t}, normal, NORMAL)' for t in TOWNS],
        'pigeonhole_witness_bound': {'N_h_max_for_at_least_one_normal_stratum': 17, 'CAL_max_for_that_stratum': 4, 'rank_at_n_CAL_4': 5},
        'proof': ['Six positive normal town counts sum to 107.', 'Two normal strata of size >=91 would require at least 182 scenarios; at most one can reach 91.',
                  'At least five normal strata therefore have N_h<=90, ceil(N_h/5)<=18, and ceil(.95*(n_CAL+1))>n_CAL.',
                  'The augmented rank selects +infinity for each such stratum, irrespective of scores/method/seed.',
                  'Q=max_h Q_h is consequently +infinity for every method/seed.',
                  'The inclusive rule 1-P(y)<=+infinity admits both labels at every valid future prediction tick, giving {normal, anomaly}.'],
        'cutoff_value': None, 'quantile_is_infinite': True, 'cutoff_computed_from_scores': False,
        'conformal_sets_created': False, 'no_protocol_repair': True}
    write('conformal_feasibility.json', feasibility)
    write('readiness.json', {**common, 'metadata_audit_complete': False, 'bounded_audit_completed_and_sealed': True,
        'ready_for_final_partition': False, 'ready_for_conformal_adapter_implementation_under_current_design': False,
        'ready_for_conformal_fitting': False, 'ready_for_scientific_evaluation': False,
        'blocking_reasons': [BLOCKER, 'Exact town/type/stratum counts and independent exclusion existence remain unverified.',
            'Historical file-level exposure ledger remains incomplete.', 'Scenario family/route/seed exchangeability remains unknown.',
            'Published aggregate counts force the unchanged global envelope to infinity.'],
        'next_step': 'Obtain an official payload-free scenario/provenance manifest. Then complete the metadata inventory, exclusion and exact stratum audit; separately authorize and seal any necessary pre-prediction metadata-only protocol amendment. Stop here.'})
    ledger = read(HERE / 'metadata_access_ledger.json')
    for entry in ledger['official_metadata_accesses']:
        if entry['state'] == 'planned_before_access':
            entry.update(state='completed_transport_selector_failed_no_selected_fields_returned',
                         result='Initial selector rejected minified unquoted heading IDs; retry recorded separately. Bytes/hash were not retained for this failed attempt.')
    ledger['web_tool_accesses'] = [
        {'order': 'before direct catalogue fetches', 'source': 'https://carlanomaly.de/download/', 'operation': 'web open publisher download page',
         'fields': ['archive names/sizes', 'catalogue links'], 'bytes': None, 'hash': None, 'reason': 'Publisher index; web tool does not return raw byte accounting', 'result': 'success'},
        {'order': 'same batch', 'source': 'https://data.carlanomaly.de/v1/', 'operation': 'web open directory index', 'bytes': 0,
         'result': 'web tool inaccessible; later direct metadata-only GET succeeded'}]
    ledger.update(completed_utc=now(), stopped_before_official_archive_or_payload=True,
                  official_archive_requests=0, official_sensor_member_reads=0, official_timestep_label_reads=0,
                  model_executions=0, conformal_fits=0, partition_executions=0,
                  limitation=BLOCKER, directory_identity_completeness=False)
    write('metadata_access_ledger.json', ledger)
    integrity()
    report()
    seal()

def integrity():
    before = read(HERE / 'preservation_snapshot.json')
    changed = [name for name, expected in before['workspace_hashes'].items() if sha(ROOT / name) != expected]
    external_changed = [name for name, expected in before['external_hashes'].items() if sha(name) != expected]
    prereg_changed = [name for name, expected in before['preregistration_hashes'].items() if sha(PRE / name) != expected]
    assert set(before['preregistration_hashes']) == {p.relative_to(PRE).as_posix() for p in PRE.rglob('*') if p.is_file()}
    old_status = set(before['git_status'].splitlines())
    new_status = set(git('status', '--porcelain=v1', '--untracked-files=all').splitlines())
    outside = [s for s in old_status ^ new_status if 'reports/carla_test_metadata_audit_v1/' not in s]
    baseline = read(PRE / 'train_dataset_integrity.json')
    stat_changes = []
    for name, expected in baseline['stats'].items():
        path = Path(name)
        assert 'train' in path.parts and 'test' not in [v.lower() for v in path.parts]
        st = path.stat()
        if {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns} != expected:
            stat_changes.append(name)
    result = {'passed': not any([changed, external_changed, prereg_changed, outside, stat_changes]) and git('rev-parse', 'HEAD') == before['git_HEAD'],
        'current_git_commit': git('rev-parse', 'HEAD'), 'changed_workspace_files': changed, 'changed_external_files': external_changed,
        'changed_preregistration_files': prereg_changed, 'status_changes_outside_audit': outside,
        'workspace_files_rehashed': len(before['workspace_hashes']), 'external_files_rehashed': len(before['external_hashes']),
        'preregistration_files_rehashed': len(before['preregistration_hashes']),
        'TRAIN_files_size_mtime_verified': len(baseline['stats']), 'TRAIN_stat_changes': stat_changes,
        'TRAIN_integrity_limit': 'Current size/mtime compared to sealed baseline following its earlier full SHA256 check. Raw TRAIN sensor files were not reopened; this is not a new full content rehash.',
        'attestation_limit': 'Action log and integrity verification, not OS-wide access monitoring.',
        'activities_performed': {'official_TEST_sensor_payload': False, 'official_TEST_timestep_labels': False, 'new_predictions': False,
            'model_inference': False, 'model_training': False, 'conformal_fit': False, 'actual_partition': False,
            'threshold_tuning': False, 'GNSS_experiment': False, 'protocol_amendment': False, 'commit_push': False}}
    write('integrity_verification.json', result)
    assert result['passed'], result

def report():
    text = f'''Metadata audit v1 — bounded audit stopped at missing official scenario metadata

Preregistration manifest: `{EXPECTED}`. Current Git commit: `0ad893172d533fc8a048a503ad681c706d5312b8`.
The current commit is the direct child of recorded pre-sealing commit `36fe79f310973321947d8d96dd78d14128cf719d`, adding only the preregistration. Frozen scientific files, seals and bindings match. No scientific revision was accepted or made.

## Metadata Access Boundary

Read publisher download/server/checksum catalogues, resource/provenance documentation, dataset section headings and selected introductory composition/directory sections. The ledger records planned sources, operations, selected fields, transferred bytes and hashes where available. Whole public documentation HTML was transported; annotation/sensor examples were discarded without interpretation. No official archive was requested, downloaded, listed or decompressed. No sensor values, image pixels, timestep labels, predictions, model inference, fit or randomized partition were accessed/executed. Existing development artifacts were read only as opaque integrity hash streams where verification required them.

## Official TEST Inventory

**Incomplete.** Publisher documentation reports 627 TEST scenarios (107 normal, 520 anomaly). These are documented aggregates, not discovered/enumerated IDs. The server index exposes archives and their checksums, with no standalone scenario manifest. Resource/code links are placeholders. The schematic directory tree does not establish actual directory existence, scenario completeness or IDs. Archive traversal would require decompression of intervening sensor/label payloads, so official access stopped. No scenario IDs, suffix ranges or families were fabricated.

## Historical Exclusion Verification

Both exact sealed identities remain permanently SCHEMA_EXPOSED_DIAGNOSTIC_ONLY and excluded from both final roles:

- `test/anomaly/Town01/change-weather/scenario-1`: complete historical exposure.
- `test/anomaly/Town01/change-weather/scenario-10`: partial historical exposure, historically recorded 93 RGB frames.

Their IDs/classifications match the amended record. Independent official metadata existence is **not verified** in this audit. No contents were reopened. The historical file-level ledger remains incomplete; no additional exposed IDs are inferred.

## Counts by Town / Condition / Type

| Quantity | Publisher aggregate | After only the two known exclusions, conditional arithmetic |
|---|---:|---:|
| Normal TEST | 107 | 107 |
| Anomaly TEST | 520 | 518 |
| Total TEST | 627 | 625 |

Both directory conditions are documented across six towns: Town01–Town05 and Town10HD. Nine anomaly types are reported. Per-town, per-type and town × condition × type counts are unavailable; JSON fields are null, never zero. The conditional eligible total does not establish complete eligibility/exposure clearance.

## Provenance / Family Audit

VERIFIED: CARLA generation and the split/town/type directory grammar. SUPPORTED BUT INCOMPLETE: a Hydra-based generation framework is described, without an accessible per-scenario mapping. UNKNOWN: same-route or simulator-seed reuse; normal/anomaly versions of a drive; weather/anomaly variants; base-drive families; explicit per-scenario provenance keys. Repeated scenario suffixes in schematic trees do not establish linkage. No reliable grouping identifier was found in inspected metadata.

## Scenario Exchangeability Assessment

Between-scenario exchangeability is not established. Latent families and route/seed duplication crossing CAL/EVAL cannot be ruled out. Town/type stratification alone does not resolve this uncertainty. If explicit linked groups are found, grouping must precede partitioning; the sealed JSON proposes connected components based only on verified keys, whole-group assignment and a separately preregistered family sampling/calibration target. Cross-stratum families may prevent exact scenario quotas. That proposal was not executed. No linked groups are claimed.

## 20/80 Count Feasibility

Frozen quota: n_CAL,h=ceil(N_h/5), n_EVAL,h=N_h−n_CAL,h. No RNG was created and no memberships/permutations were produced. Exact quota simulation by stratum is blocked by missing counts. For six positive normal town counts totaling 107, aggregate normal CAL counts can range from 22 to 26 and normal EVAL from 81 to 85. These are attainable algebraic bounds, not realized allocations. Anomaly and total quotas remain unknown.

## Conformal Stratum Adequacy

At alpha=.05, k=ceil(19(n_CAL+1)/20), using the augmented +infinity order statistic. A finite rank requires n_CAL>=19, hence N_h>=91 under the frozen quota. Six normal strata sum to 107; at most one can reach 91. **At least five normal town strata are necessarily sparse under the publisher counts.** At least one has N_h<=floor(107/6)=17 and n_CAL<=4. Its k>n_CAL selects +infinity. Exact sparse town identities and counts cannot be determined from available metadata, so none are asserted. Anomaly strata cannot be audited exactly.

## Global Envelope Feasibility

**Conditional on the publisher aggregates and documented town stratification, Q=max_h Q_h must equal +infinity for all three methods and five seeds.** A single sparse required stratum suffices, regardless of predictions or score values. The future inclusive candidate rule admits both labels at every valid prediction tick, yielding {{normal, anomaly}} everywhere. No sets, scores or cutoffs were fitted/computed from model outputs. This is count/rank mathematics, not an observed performance result. Exact culprit strata remain unavailable.

## Protocol Issues Requiring Amendment

The frozen 20/80 quota and global envelope are necessarily vacuous under published counts. No alpha, strata, ratio, pooling, score, envelope, seed, metric or threshold was changed. A correction requires separate authorization and a sealed metadata-only pre-prediction amendment. Resolve official inventory/provenance and historical exposure gates first; do not choose a repair using model results. Group-level changes may also be required if verified linkage emerges.

## Ready / Not Ready for Final Partition

**Not ready.** Complete scenario identities/counts and independent historical exclusion existence are missing; the historical file-level ledger is incomplete; family exchangeability is unknown; the frozen design is vacuous under documented counts. Actual allocation remains prohibited.

## Ready / Not Ready for Conformal Adapter Implementation

**Not ready under the current design.** An adapter could mechanically preserve infinity, but implementing a corrected final design must wait for the missing metadata audit and separately sealed amendment. No adapter or generic COGNIX code was modified.

## Recommended Next Step

Obtain a publisher-issued scenario-only manifest containing full TEST paths, completeness, town, directory condition, anomaly type and explicit route/seed/base-drive keys if available, without sensor values or timestep/event annotations. Complete exact inventory/exclusion/count/provenance checks, then separately authorize and seal any mathematical/provenance correction before prediction/payload access. Stop here.

Integrity: all 3,764 existing workspace files, 36 external bindings and 57 preregistration files were rehashed unchanged; all 15 run/result seals and graph/protocol/environment bindings passed. All 120,100 TRAIN files retain their sealed size/mtime baseline; no new raw TRAIN content hash was performed. Current Git and tracked working tree remain unchanged; changes are confined to this new audit directory. No commit/push.

Sources: [Publisher composition and directory documentation](https://carlanomaly.de/dataset/), [download catalogue](https://carlanomaly.de/download/), [server directory index](https://data.carlanomaly.de/v1/), [checksum catalogue](https://data.carlanomaly.de/v1/SHA256SUMS.txt), [resource catalogue](https://carlanomaly.de/resources/), [project provenance FAQ](https://carlanomaly.de/about/). Selected source fields and hashes are preserved in the audit. Publisher documentation may change; no archive bytes were compared to the listed archive checksums.
'''
    (HERE / 'report.md').write_text(text, encoding='utf-8')

def seal():
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS', 'SHA256SUMS.sha256'])
    manifest = HERE / 'SHA256SUMS'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(HERE).as_posix()}\n' for p in paths), encoding='utf-8')
    (HERE / 'SHA256SUMS.sha256').write_text(f'{sha(manifest)}  SHA256SUMS\n', encoding='utf-8')
    for line in manifest.read_text().splitlines():
        expected, rel = line.split('  ', 1)
        assert sha(HERE / rel) == expected
    for p in HERE.glob('*.json'):
        json.loads(p.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    assert not read(HERE / 'readiness.json')['ready_for_final_partition']
    assert not read(HERE / 'partition_feasibility.json')['actual_partition_executed']
    print(json.dumps({'sealed_files': len(paths), 'manifest_sha256': sha(manifest), 'integrity_passed': True,
                      'audit_complete': False, 'global_infinity_conditional_on_published_counts': True}))

if __name__ == '__main__':
    main()
