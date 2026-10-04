"""Amendment 003: prepare without RNG, execute once, verify without RNG.

Never retry execute after its receipt exists, including an incomplete receipt.
Only committed structure metadata is read; no TEST archive/payload access.
"""
import ast
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SCRIPT = Path(__file__).resolve()
ARTIFACT = 'reports/carla_test_structure_inventory_v1/kaggle_result_v1/cognix_carla_test_inventory_result_v1.zip'
PREFIX = 'cognix_carla_test_inventory_result_v1/'
HEAD = '836a9245f72d7b6a53647714954ec50c8052b620'
ZIP_HASH = '6f1bfa9bfb6a2401263c4b0642243f779632cf79c7f0b6cd7f07f95aaa6999b3'
SOURCE_SEAL = 'e5219b9dae9447bf8d818828e5ef9ee7987cc5d375f345c4be7d52ad045a333c'
ELIGIBLE_SEAL = '9c08a8cc404b857d8030d316eac12f26fd9145e301ecdce1c081a4c82e981218'
SEED = 334227055836169582741732795999208677637
EXCLUSIONS = {'test/anomaly/Town01/change-weather/scenario-1', 'test/anomaly/Town01/change-weather/scenario-10'}
CAL, EVAL = 'FINAL_CONFORMAL_CAL', 'FINAL_EVALUATION'
REQUIRED = {'pre_execution_verification.json', 'partition_execution_receipt.json',
            'final_conformal_cal.txt', 'final_evaluation.txt', 'final_conformal_cal.json',
            'final_evaluation.json', 'partition_membership.json', 'descriptive_counts.json',
            'partition_validation.json', 'report.md', 'SHA256SUMS', 'SHA256SUMS.sha256',
            'execute_partition.py'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def stamp():
    now = datetime.now(timezone.utc)
    return {'timestamp_utc': now.isoformat(), 'timestamp_Asia_Calcutta':
            now.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat()}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def write(name, data, exclusive=True):
    with (OUT / name).open('xb' if exclusive else 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    require((OUT / name).read_bytes() == data, 'Output read-back failed: ' + name)


def outside_status():
    entries = git('status', '--porcelain=v1', '-z', '--untracked-files=all').split(b'\0')
    return b'\0'.join(e for e in entries if e and not e[3:].startswith(b'reports/carla_final_partition_v1/'))


def check_manifest(files, name, expected):
    found = {}
    for line in files[name].decode('utf-8').splitlines():
        h, n = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', h) and n in files and n not in found,
                'Malformed/duplicate manifest entry: ' + name)
        require(digest(files[n]) == h, 'Manifest hash failure: ' + n)
        found[n] = h
    require(set(found) == set(expected), 'Manifest coverage failure: ' + name)
    return found


def composition(ids):
    conditions, towns, types = Counter(), Counter(), Counter()
    for scenario in ids:
        parts = scenario.split('/')
        conditions[parts[1]] += 1
        towns[parts[2]] += 1
        types[parts[3] if parts[1] == 'anomaly' else 'NORMAL'] += 1
    return {'total': len(ids), 'normal_anomaly': dict(sorted(conditions.items())),
            'town': dict(sorted(towns.items())), 'anomaly_type': dict(sorted(types.items()))}


def verify_input():
    require(OUT == ROOT / 'reports/carla_final_partition_v1', 'Wrong output directory')
    require(git('rev-parse', 'HEAD').decode().strip() == HEAD, 'HEAD changed')
    require(git('branch', '--show-current').decode().strip() == 'main', 'Branch changed')
    require(git('rev-parse', 'myfork/main').decode().strip() == HEAD, 'myfork/main changed')
    require(not git('status', '--porcelain', '-uno').strip(), 'Tracked changes exist')
    blob = git('show', 'HEAD:' + ARTIFACT)
    require(digest(blob) == ZIP_HASH and (ROOT / ARTIFACT).read_bytes() == blob,
            'Authoritative ZIP differs from committed bytes')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        require(all(n.startswith(PREFIX) and '/' not in n[len(PREFIX):] for n in names),
                'Unexpected ZIP paths')
        require(archive.testzip() is None, 'ZIP CRC failure')
        files = {n[len(PREFIX):]: archive.read(n) for n in names}
    verified = {}
    for name, expected in (
            ('SHA256SUMS', set(files) - {'SHA256SUMS', 'SHA256SUMS.sha256'}),
            ('SOURCE_INVENTORY_SHA256SUMS', {'scenario_inventory.json', 'scenario_inventory_sorted.txt'}),
            ('ELIGIBLE_INVENTORY_SHA256SUMS', {'eligible_inventory.json', 'eligible_inventory_sorted.txt', 'historical_exclusions.json'})):
        verified[name] = check_manifest(files, name, expected)
        check_manifest(files, name + '.sha256', {name})
    require(digest(files['SOURCE_INVENTORY_SHA256SUMS']) == SOURCE_SEAL, 'Source seal mismatch')
    require(digest(files['ELIGIBLE_INVENTORY_SHA256SUMS']) == ELIGIBLE_SEAL, 'Eligible seal mismatch')
    readiness = json.loads(files['readiness.json'])
    require(readiness['status'] == 'READY_FOR_PARTITION_EXECUTION' and readiness['inventory_valid'] is True,
            'Inventory not ready/valid')
    require(readiness['source_inventory_seal'] == SOURCE_SEAL and readiness['eligible_inventory_seal'] == ELIGIBLE_SEAL,
            'Readiness seals mismatch')
    source, eligible = json.loads(files['scenario_inventory.json']), json.loads(files['eligible_inventory.json'])
    for rows, count, txt in ((source, 627, 'scenario_inventory_sorted.txt'), (eligible, 625, 'eligible_inventory_sorted.txt')):
        ids = [r['scenario_id'] for r in rows]
        require(len(ids) == len(set(ids)) == count, 'Inventory count/uniqueness failure')
        for row in rows:
            sid = row['scenario_id']
            require(re.fullmatch(r'test/(normal/Town[0-9]+(?:HD)?/scenario-[1-9][0-9]*|anomaly/Town[0-9]+(?:HD)?/[a-z][a-z-]*/scenario-[1-9][0-9]*)', sid), 'Malformed canonical ID')
            parts = sid.split('/')
            require(row['condition'] == parts[1] and row['town'] == parts[2] and
                    row['anomaly_type'] == (parts[3] if parts[1] == 'anomaly' else 'NORMAL'), 'Path metadata mismatch')
            require(row['structural_completion_state'] == 'COMPLETE_AT_VERIFIED_ARCHIVE_EOF', 'Incomplete scenario')
        require(sorted(ids) == sorted(ids, key=lambda s: s.encode('utf-8')), 'Bytewise sorting mismatch')
        require(files[txt] == ('\n'.join(sorted(ids)) + '\n').encode('utf-8'), 'Sorted inventory representation mismatch')
    source_ids, eligible_ids = {r['scenario_id'] for r in source}, {r['scenario_id'] for r in eligible}
    require(source_ids - eligible_ids == EXCLUSIONS and eligible_ids == source_ids - EXCLUSIONS,
            'Exact historical exclusion reconciliation failed')
    source_by_id = {r['scenario_id']: r for r in source}
    require(all(row == source_by_id[row['scenario_id']] for row in eligible), 'Eligible metadata differs from source')
    require(composition(list(source_ids))['normal_anomaly'] == {'normal': 107, 'anomaly': 520}, 'Source composition mismatch')
    require(composition(list(eligible_ids))['normal_anomaly'] == {'normal': 107, 'anomaly': 518}, 'Eligible composition mismatch')
    exclusions = json.loads(files['historical_exclusions.json'])
    require(set(exclusions['exact_ids']) == set(exclusions['removed_exactly']) == EXCLUSIONS and
            len(exclusions['exact_ids']) == len(exclusions['removed_exactly']) == 2 and
            exclusions['both_present'] is True and exclusions['additional_exposure_inferred'] is False,
            'Historical exclusion record mismatch')
    reconciliation = json.loads(files['count_reconciliation.json'])
    expected_source, expected_eligible = {'total': 627, 'normal': 107, 'anomaly': 520}, {'total': 625, 'normal': 107, 'anomaly': 518}
    require(reconciliation['passed'] is True and reconciliation['observed_root_counts'] == reconciliation['expected_source'] == expected_source and
            reconciliation['eligible'] == reconciliation['expected_eligible'] == expected_eligible, 'Counts record mismatch')
    structure = json.loads(files['structure_validation.json'])
    require(structure['passed'] is True and structure['failure_reason'] is None and all(v is True for v in structure['gates'].values()) and
            structure['source_inventory_seal'] == SOURCE_SEAL and structure['eligible_inventory_seal'] == ELIGIBLE_SEAL, 'Structure gates failed')
    for filename, field in (('town_counts.json', 'town'), ('anomaly_type_counts.json', 'anomaly_type')):
        require(json.loads(files[filename])['counts'] == composition(list(source_ids))[field], 'Descriptive source counts mismatch')
    archive_record = json.loads(files['source_archive_verification.json'])
    require(archive_record['full_archive_verified'] is True and archive_record['gzip_EOF'] is True and archive_record['tar_EOF'] is True and
            archive_record['expected_SHA256'] == archive_record['received_SHA256'] and archive_record['expected_bytes'] == archive_record['received_bytes'], 'Archive verification record failed')
    protocol_path = 'reports/carla_final_evaluation_preregistration_v1/'
    protocol_blob = git('show', 'HEAD:' + protocol_path + 'test_partition_protocol.json')
    protocol = json.loads(protocol_blob)
    seed_blob = git('show', 'HEAD:' + protocol_path + 'partition_randomization_receipt.json')
    require(digest(seed_blob) == '2a6cc338637a6a2b6ce661f6711ac4ea3e45ff6337346d27ff1b4753e3377368' == protocol['seed_receipt']['sha256'], 'Seed receipt binding failed')
    require(json.loads(seed_blob)['seed'] == protocol['seed'] == protocol['allocation']['seed'] == SEED,
            'Frozen seed mismatch')
    require(protocol['amendment_id'] == '003_partition_randomization_clarification_only' and
            protocol['allocation']['permutation_calls'] == 1 and protocol['allocation']['rng'] == 'PCG64' and
            protocol['allocation']['stratify_assignment'] is False and protocol['allocation']['redraws_or_balancing'] is False and
            set(protocol['exact_excluded_scenario_ids']) == EXCLUSIONS, 'Amendment 003 algorithm mismatch')
    ids = sorted(eligible_ids)
    require(len(ids) == len(set(ids)) == 625 and not EXCLUSIONS.intersection(ids), 'Eligible input failed')
    require(math.ceil(625 / 5) == 125, 'Frozen CAL size failed')
    return ids, {'git_HEAD': HEAD, 'git_branch': 'main', 'git_myfork_main': HEAD,
                 'authoritative_committed_inventory_artifact': ARTIFACT, 'ZIP_sha256': ZIP_HASH,
                 'source_inventory_seal': SOURCE_SEAL, 'eligible_inventory_seal': ELIGIBLE_SEAL,
                 'readiness': readiness, 'source_counts': expected_source, 'eligible_counts': expected_eligible,
                 'historical_exclusions': sorted(EXCLUSIONS), 'historical_exclusions_absent_from_eligible': True,
                 'eligible_count': 625, 'internal_manifests_verified': verified,
                 'sorted_input_representation': 'UTF-8 IDs joined by LF with one trailing LF',
                 'sorted_input_sha256': digest(files['eligible_inventory_sorted.txt']),
                 'algorithm': 'NumPy Generator(PCG64(seed))', 'seed': SEED, 'seed_decimal': str(SEED),
                 'sorting': 'lexicographic before permutation; verified UTF-8 bytewise equivalent',
                 'expected_CAL_size': 125, 'expected_EVAL_size': 500,
                 'stratification': False, 'balancing': False, 'redraw': False,
                 'numpy_version': np.__version__, 'python_version': sys.version,
                 'protocol_sha256': digest(protocol_blob), 'sealed_seed_receipt_sha256': digest(seed_blob),
                 'execution_script_sha256': digest(SCRIPT.read_bytes()), 'rng_instantiations_at_verification': 0,
                 'permutation_calls_at_verification': 0, 'outside_output_git_status_sha256': digest(outside_status())}


def seal():
    files = {p.name: p.read_bytes() for p in OUT.iterdir() if p.is_file() and p.name not in {'SHA256SUMS', 'SHA256SUMS.sha256'}}
    manifest = ''.join(f'{digest(files[n])}  {n}\n' for n in sorted(files)).encode('utf-8')
    write('SHA256SUMS', manifest, exclusive=False)
    write('SHA256SUMS.sha256', f'{digest(manifest)}  SHA256SUMS\n'.encode(), exclusive=False)
    return digest(manifest)


def prepare():
    require({p.name for p in OUT.iterdir()} == {'execute_partition.py'}, 'Output exists: refuse preparation/re-execution')
    ids, pre = verify_input()
    calls = [n.func for n in ast.walk(ast.parse(SCRIPT.read_text(encoding='utf-8'))) if isinstance(n, ast.Call)]
    for attribute in ('PCG64', 'Generator', 'permutation'):
        require(sum(isinstance(c, ast.Attribute) and c.attr == attribute for c in calls) == 1, 'Static RNG call count failed')
    pre.update(stamp())
    pre.update({'status': 'PRE_EXECUTION_VERIFIED_AND_SEALED', 'static_rng_call_sites': 1,
                'static_permutation_call_sites': 1, 'authorization': 'Explicit user one-time execution request'})
    write('pre_execution_verification.json', encoded(pre))
    print('PRE_EXECUTION_SEALED', seal(), 'RNG instantiations: 0; permutations: 0')


def verify_pre_seal():
    files = {p.name: p.read_bytes() for p in OUT.iterdir() if p.is_file()}
    check_manifest(files, 'SHA256SUMS', {'pre_execution_verification.json', 'execute_partition.py'})
    check_manifest(files, 'SHA256SUMS.sha256', {'SHA256SUMS'})


def verify_membership(ids, files, receipt):
    cal = files['final_conformal_cal.txt'].decode('utf-8').splitlines()
    evaluation = files['final_evaluation.txt'].decode('utf-8').splitlines()
    require(json.loads(files['final_conformal_cal.json']) == cal and json.loads(files['final_evaluation.json']) == evaluation, 'Role JSON/TXT mismatch')
    union = set(cal) | set(evaluation)
    require(len(cal) == len(set(cal)) == 125 and len(evaluation) == len(set(evaluation)) == 500, 'Role count/uniqueness failed')
    require(not set(cal).intersection(evaluation) and union == set(ids) and len(union) == 625 and not EXCLUSIONS.intersection(union), 'Partition set validation failed')
    ordered = cal + evaluation
    require(receipt['ordered_permutation'] == ordered, 'Persisted permutation differs from roles')
    membership = json.loads(files['partition_membership.json'])
    expected_rows = [{'scenario_id': sid, 'permutation_position_0_based': i, 'role': CAL if i < 125 else EVAL} for i, sid in enumerate(ordered)]
    require(membership['ordered_assignment'] == expected_rows, 'Ordered assignment mismatch')
    require(receipt['membership_file_sha256'] == {n: digest(files[n]) for n in receipt['membership_file_sha256']}, 'Membership hash mismatch')
    return {'status': 'PASSED', 'deterministic_only': True, 'no_RNG_created_or_permutation_replayed_for_validation': True,
            'CAL_count': len(cal), 'EVAL_count': len(evaluation), 'CAL_unique_count': len(set(cal)),
            'EVAL_unique_count': len(set(evaluation)), 'overlap_count': 0, 'union_count': len(union),
            'union_equals_exact_eligible_set': True, 'historical_exclusion_intersection_count': 0,
            'ordered_permutation_and_assignment_verified': True, 'role_JSON_TXT_equivalent': True,
            'membership_hashes_verified': True}


def execute():
    require({p.name for p in OUT.iterdir()} == {'execute_partition.py', 'pre_execution_verification.json', 'SHA256SUMS', 'SHA256SUMS.sha256'},
            'Unexpected output/receipt exists: STOP; never retry randomization')
    verify_pre_seal()
    ids, fresh = verify_input()
    pre = json.loads((OUT / 'pre_execution_verification.json').read_bytes())
    require(all(pre[k] == v for k, v in fresh.items()), 'Pre-execution bindings changed')
    receipt = {'status': 'EXECUTION_ARMED_DO_NOT_RETRY', 'seed': SEED, 'seed_decimal': str(SEED),
               'seed_permanently_spent': True, 'bit_generator': 'PCG64', 'numpy_version': np.__version__,
               'python_version': sys.version, 'input_eligible_inventory_seal': ELIGIBLE_SEAL,
               'sorted_input_sha256': pre['sorted_input_sha256'], 'input_count': 625,
               'pre_execution_verification_sha256': digest((OUT / 'pre_execution_verification.json').read_bytes()),
               'pre_execution_seal': digest((OUT / 'SHA256SUMS').read_bytes()),
               'retry_prohibited': True, 'stratification': False, 'balancing': False, 'redraw': False,
               'rng_instantiations': 0, 'permutation_calls': 0, **stamp()}
    write('partition_execution_receipt.json', encoded(receipt))
    ordered = None
    try:
        rng = np.random.Generator(np.random.PCG64(334227055836169582741732795999208677637))
        receipt['rng_instantiations'] = 1
        ordered = rng.permutation(ids).tolist()
        receipt['permutation_calls'] = 1
        receipt.update(stamp())
        n_cal = math.ceil(625 / 5)
        cal, evaluation = ordered[0:n_cal], ordered[n_cal:625]
        membership = {'unit': 'whole scenario', 'input_eligible_inventory_seal': ELIGIBLE_SEAL,
                      'sorted_input_sha256': pre['sorted_input_sha256'],
                      'ordered_assignment': [{'scenario_id': sid, 'permutation_position_0_based': i,
                                              'role': CAL if i < n_cal else EVAL} for i, sid in enumerate(ordered)]}
        outputs = {'final_conformal_cal.txt': ('\n'.join(cal) + '\n').encode('utf-8'),
                   'final_evaluation.txt': ('\n'.join(evaluation) + '\n').encode('utf-8'),
                   'final_conformal_cal.json': encoded(cal), 'final_evaluation.json': encoded(evaluation),
                   'partition_membership.json': encoded(membership)}
        receipt.update({'status': 'PERMUTATION_RECORDED_SEED_PERMANENTLY_SPENT', 'ordered_permutation': ordered,
                        'n_cal': n_cal, 'CAL_count': len(cal), 'EVAL_count': len(evaluation),
                        'overlap_count': len(set(cal) & set(evaluation)), 'union_count': len(set(ordered)),
                        'historical_exclusion_intersection_count': len(EXCLUSIONS.intersection(ordered)),
                        'membership_file_sha256': {n: digest(b) for n, b in outputs.items()}})
        # Persist the recoverable complete permutation and all membership hashes first.
        write('partition_execution_receipt.json', encoded(receipt), exclusive=False)
        for name, data in outputs.items():
            write(name, data)
        validation = verify_membership(ids, {n: (OUT / n).read_bytes() for n in outputs}, receipt)
        counts = {'observational_only': True, 'membership_reassigned': False,
                  CAL: composition(cal), EVAL: composition(evaluation)}
        write('descriptive_counts.json', encoded(counts))
        write('partition_validation.json', encoded({**validation, **stamp()}))
        report = ('# COGNIX final CARLA partition v1\n\nFINAL_PARTITION_SEALED\n\n'
                  f'Amendment 003 executed once with NumPy {np.__version__}, PCG64, seed {SEED}. '
                  'The seed is permanently spent. One Generator/PCG64 instantiation and one permutation; '
                  'no stratification, balancing, redraw or reassignment.\n\n'
                  'The committed structure-only inventory passed independent manifest, hash, readiness, '
                  'count and seal checks before any RNG was instantiated. The pre-execution JSON was sealed '
                  'before execution; its hash and initial manifest seal are retained in the execution receipt.\n\n'
                  f'Source inventory seal: `{SOURCE_SEAL}`.\n\nEligible inventory seal: `{ELIGIBLE_SEAL}`.\n\n'
                  'Sorted input is the exact UTF-8/LF representation in eligible_inventory_sorted.txt; '
                  f'its SHA-256 is `{pre["sorted_input_sha256"]}`.\n\n'
                  'Permutation order and zero-based assignment positions are preserved. Positions [0:125] '
                  'are FINAL_CONFORMAL_CAL (125); [125:625] are FINAL_EVALUATION (500). '
                  'Overlap is zero; union is exactly the 625 eligible IDs; both historical exclusions are absent.\n\n'
                  f'Membership SHA-256: `{digest(outputs["partition_membership.json"])}`.\n\n'
                  'The aggregate partition seal is the SHA-256 of SHA256SUMS, recorded in SHA256SUMS.sha256. '
                  'The manifest covers this script and all ten JSON/TXT/Markdown outputs; seal files do not hash themselves.\n\n'
                  'Descriptive counts below are observational only, derived from canonical ID paths. '
                  'NORMAL is the normal-scenario anomaly-type bucket. No composition affected assignment.\n\n'
                  '```json\n' + encoded(counts).decode('utf-8') + '```\n\n'
                  'Validation used only persisted lists, sets and hashes; randomization was never replayed. '
                  'No TEST archive payload, images, Feather contents, anomaly labels/timesteps, sensor values, '
                  'features, models, predictions, conformal fitting or performance metrics were accessed. '
                  'No source inventory or preregistration file was modified; no commit or push was performed.\n')
        write('report.md', report.encode('utf-8'))
        seal()
    except BaseException as error:
        receipt['failure_after_execution_armed'] = repr(error)
        receipt['recovery_rule'] = 'STOP. Preserve evidence. Never rerun randomization.'
        if ordered is not None:
            receipt['ordered_permutation'] = ordered
        write('partition_execution_receipt.json', encoded(receipt), exclusive=False)
        raise
    verify_outputs()


def verify_outputs():
    require({p.name for p in OUT.iterdir()} == REQUIRED, 'Output file set differs from required set')
    ids, fresh = verify_input()
    files = {p.name: p.read_bytes() for p in OUT.iterdir()}
    check_manifest(files, 'SHA256SUMS', REQUIRED - {'SHA256SUMS', 'SHA256SUMS.sha256'})
    check_manifest(files, 'SHA256SUMS.sha256', {'SHA256SUMS'})
    pre, receipt = json.loads(files['pre_execution_verification.json']), json.loads(files['partition_execution_receipt.json'])
    require(all(pre[k] == v for k, v in fresh.items()), 'Environment/input changed after partition')
    require(receipt['pre_execution_verification_sha256'] == digest(files['pre_execution_verification.json']), 'Pre-execution hash mismatch')
    initial_manifest = ''.join(f'{digest(files[n])}  {n}\n' for n in sorted({'execute_partition.py', 'pre_execution_verification.json'})).encode()
    require(receipt['pre_execution_seal'] == digest(initial_manifest), 'Pre-execution seal mismatch')
    require(receipt['status'] == 'PERMUTATION_RECORDED_SEED_PERMANENTLY_SPENT' and receipt['seed_permanently_spent'] is True and
            receipt['rng_instantiations'] == receipt['permutation_calls'] == 1 and receipt['seed'] == SEED and
            receipt['bit_generator'] == 'PCG64' and receipt['input_eligible_inventory_seal'] == ELIGIBLE_SEAL and
            receipt['sorted_input_sha256'] == pre['sorted_input_sha256'] and receipt['input_count'] == 625 and receipt['n_cal'] == 125,
            'Execution receipt invalid')
    validation = verify_membership(ids, files, receipt)
    require(all(json.loads(files['partition_validation.json'])[k] == v for k, v in validation.items()), 'Persisted validation mismatch')
    for key in ('CAL_count', 'EVAL_count', 'overlap_count', 'union_count', 'historical_exclusion_intersection_count'):
        require(receipt[key] == validation[key], 'Receipt validation mismatch')
    counts = json.loads(files['descriptive_counts.json'])
    require(counts == {'observational_only': True, 'membership_reassigned': False,
                      CAL: composition(json.loads(files['final_conformal_cal.json'])),
                      EVAL: composition(json.loads(files['final_evaluation.json']))}, 'Descriptive counts mismatch')
    print('FINAL_PARTITION_SEALED')
    print('CAL count: 125; EVAL count: 500; deterministic validation: PASSED')
    print('Partition seal/SHA256SUMS SHA-256:', digest(files['SHA256SUMS']))
    print('Seed permanently spent: true; RNG instantiations: 1; permutation calls: 1')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in {'prepare', 'execute', 'verify'}, 'Specify prepare, execute, or verify')
    {'prepare': prepare, 'execute': execute, 'verify': verify_outputs}[sys.argv[1]]()
