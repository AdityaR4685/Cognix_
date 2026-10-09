"""Future authorized TRAIN-only Gate-2. Both PASS and FAIL stop for human review."""
import argparse
import json
import sys
from gate2_common import (BUNDLE, TARGET, TOKEN, MODALITIES, require, install_guard,
                          verify_seal, write_new, pretty, canonical, hash_file, read_json)
from gate2_preflight import verify_inputs, progress_printer, environment
from gate2_science import load_science, new_agent, member_arrays, restore_agent
from gate2_data import assemble_fit, npz_bytes, load_npz, modality_calibration, array_digest
from gate2_audit import audit_modality, gate_decision
from gate2_runtime import Runtime
from gate2_pseudo import prepare_cal_pseudo, read_pseudo, write_jsonl_gzip


def authorization(token):
    require(token == TOKEN, 'Exact human authorization token required before runtime creation or fitting')


def fit_unit(runtime, science, modality):
    name = 'fit-' + modality
    if runtime.has(name):
        path = runtime.unit(name)
        arrays = {}
        for i in range(5):
            arrays.update({f'member_{i}_{key}': value for key, value in load_npz(path / f'member_{i}.npz').items()})
        return restore_agent(science, modality, arrays), read_json(path / 'fit_binding.json')
    x, binding = assemble_fit(runtime.evidence['membership'], modality)
    require(len(binding['ordered_consumed_scenarios']) == 76 and binding['total_rows'] == 76 * 2999,
            'All 76 FIT clean scenarios required')
    agent = new_agent(science, modality)
    def writer(pending):
        write_new(pending / 'fit_binding.json', pretty(binding))
        agent.fit(x)  # exact unchanged production one-class/bootstrap implementation
        arrays = member_arrays(agent)
        import numpy as np
        require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite upstream fitted state')
        hashes = []
        for i in range(5):
            state = {key: arrays[f'member_{i}_{key}'] for key in ('mean', 'precision', 'd2_scale', 'shrinkage')}
            write_new(pending / f'member_{i}.npz', npz_bytes(state))
            hashes.append({'member': i, 'state_file_sha256': hash_file(pending / f'member_{i}.npz'),
                           'state_content_sha256': array_digest(state)})
        write_new(pending / 'upstream_state_hashes.json', pretty(hashes))
    runtime.commit(name, writer, {'fit_input_sha256': binding['input_array_sha256'],
                   'partition_sha256': runtime.manifest['partition_sha256']})
    return agent, binding


def execute(bundle_seal, token):
    authorization(token)
    counters = install_guard(writable=(TARGET,), allow_source=True)
    verify_seal(BUNDLE, bundle_seal)
    evidence = verify_inputs(progress_printer())  # independent full verification BEFORE any fitting
    runtime = Runtime(TARGET, bundle_seal, evidence)
    state = runtime.verify()['state']
    if state == 'complete_verified_immutable':
        return dict(read_json(TARGET / 'FINAL.json'), runtime_state=state,
                    runtime_seal=hash_file(TARGET / 'SHA256SUMS'))
    science = load_science()
    phase = 'authorized_verified_runtime'
    with runtime.lease():
        if (TARGET / 'FINAL.json').exists():
            return runtime.finalize(read_json(TARGET / 'FINAL.json')['modality_status'])
        runtime.begin_attempt()
        try:
            agents, fits = {}, {}
            for modality in MODALITIES:
                phase = 'FIT_clean_only_' + modality
                print('Future upstream fitting/reuse: ' + modality, flush=True)
                agents[modality], fits[modality] = fit_unit(runtime, science, modality)
            phase = 'CAL_pseudo_only'
            prepare_cal_pseudo(runtime, science)
            statuses = {}
            for modality in MODALITIES:
                phase = 'CAL_mapping_and_audit_' + modality
                name = 'audit-' + modality
                if not runtime.has(name):
                    features, labels, rows = modality_calibration(evidence['membership'], modality,
                        lambda sid: read_pseudo(runtime.unit('cal-' + sid.replace('/', '__'))))
                    print('Future CAL mapping and complete audit: ' + modality, flush=True)
                    # The independent modalities are audited even after another fails.
                    # No invalid channel escapes this audit-only evidence namespace.
                    def writer(pending):
                        write_jsonl_gzip(pending / 'calibration_provenance.jsonl.gz', rows)
                        report, arrays = audit_modality(science, agents[modality], modality,
                                                       features, labels, rows, fits[modality], evidence['membership'])
                        report.update(scientific_hashes=evidence['bindings']['scientific_hashes'],
                                      source_manifest_sha256=hash_file(TARGET / 'run_manifest.json'))
                        write_new(pending / 'audit_arrays.npz', npz_bytes(arrays))
                        write_new(pending / 'audit.json', pretty(report))
                    runtime.commit(name, writer, {'FIT_unit_seal': verify_seal(runtime.unit('fit-' + modality))['seal'],
                        'CAL_unit_seals': {sid: verify_seal(runtime.unit('cal-' + sid.replace('/', '__')))['seal']
                                          for sid in sorted(evidence['membership']['CAL_NORMAL'])},
                        'source_replay_seal': verify_seal(runtime.unit('cal-source-verification'))['seal']})
                statuses[modality] = read_json(runtime.unit(name) / 'audit.json')['status']
            phase = 'final_train_only_decision'
            status = gate_decision(statuses)
            def decision_writer(pending):
                write_new(pending / 'gate2_decision.json', pretty({'status': status,
                    'modality_status': statuses, 'action': 'STOP FOR HUMAN REVIEW',
                    'graph_constructed': False, 'GAT_executed': False,
                    'TEST_requests': 0, 'network_requests': 0}))
                text = ('# Experiment 2B Gate-2 methodology decision\n\n' + status + '\n\n' +
                        '\n'.join(m + ': ' + statuses[m] for m in MODALITIES) + '\n\n' +
                        'The evidence concerns P(target = normal) under the TRAIN-derived constructed clean-vs-pseudo calibration task. '
                        'Pseudo discrimination is a TRAIN development diagnostic only. Natural-log entropy nats and zero/tiny epistemic values are retained.\n\n' +
                        ('All active channels passed the frozen scientific-health predicates. Separate human review and separate graph authorization are required.\n\n'
                         if status.endswith('PASS') else
                         'At least one active channel is invalid. Stop before graph construction. Preserve every audit and failure record. No automatic repair is permitted. Any new scientific repair requires separately preregistered Experiment 2C or later.\n\n') +
                        'STOP FOR HUMAN REVIEW\n')
                write_new(pending / 'methodology_decision.md', text.encode())
            require(all(v == 0 for v in counters.values()), 'Execution guard violation')
            require(environment() == evidence['environment'], 'Environment changed during execution')
            # Verify every immutable upstream evidence file again before finalization.
            from partition_common import verify_preservation
            verify_preservation(read_json(BUNDLE / 'preservation_baseline.json'))
            runtime.commit('decision', decision_writer, {'audit_unit_seals': {
                m: verify_seal(runtime.unit('audit-' + m))['seal'] for m in MODALITIES}})
            runtime.finish_attempt('COMPLETE_TRAIN_ONLY_GATE2', phase)
            return dict(runtime.finalize(statuses), guard_counters=counters)
        except BaseException as exc:
            if not (runtime.attempt / 'SHA256SUMS.sha256').exists():
                runtime.finish_attempt('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'EXECUTION_STOPPED', phase, exc)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal', required=True)
    parser.add_argument('--authorize-gate2', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.bundle_seal, args.authorize_gate2), indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        print('STOP FOR HUMAN REVIEW: ' + repr(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
