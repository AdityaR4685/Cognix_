"""Render already sealed values and record the verified local delivery; no training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINEERING = ROOT / 'reports/carla_gat_paired_execution_v1'
RESULTS = ROOT / 'reports/carla_gat_paired_runs_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


verification = read(ENGINEERING / 'local_import_verification.json')
assert verification['passed'] and len(verification['all_15_runs']) == 15
assert not verification['changed_preexisting_files']
run_data = read(RESULTS / 'per_seed_metrics.json')['runs']
scenario_data = read(RESULTS / 'scenario_metrics.json')['runs']
recipe_data = read(RESULTS / 'recipe_metrics.json')['runs']
order = [f'{method}_{seed}' for seed in (101, 202, 303, 404, 505)
         for method in ('nograph', 'standard_gat', 'epistemic_gat')]
metrics = ('AUROC', 'AUPRC', 'F1', 'accuracy', 'balanced_accuracy', 'Brier', 'BCE', 'ECE')
display = {'nograph': 'NoGraph', 'standard_gat': 'StandardGAT', 'epistemic_gat': 'EpistemicGAT'}


def number(value):
    return 'null' if value is None else format(value, '.9f')


def table(rows, prefix):
    header = prefix + list(metrics)
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    for label, values in rows:
        lines.append('| ' + ' | '.join(label + [number(values.get(k)) for k in metrics]) + ' |')
    return '\n'.join(lines) + '\n'


report = (RESULTS / 'report.md').read_text(encoding='utf-8')
report = report.replace('All 15 preregistered real graph runs completed',
    'Local delivery is complete. The downloaded archive matches its Kaggle SHA-256; '
    '669 sealed files and 426 checkpoint internal hashes passed verification. '
    'All 15 prediction/metric identities and frozen statistical recomputations passed. '
    'The original sealed Kaggle report and results remain unchanged. This delivery report '
    'renders their recorded values; nine-decimal table rounding is for display only.\n\n'
    'All 15 preregistered real graph runs completed', 1)
report = report.replace('Local input/code/upstream preservation is verified when downloaded evidence is imported.',
    'Local import verified all 1,691 protected pre-existing files, 30 upstream hashes, '
    'seven graph files, Git HEAD and the exact environment lock unchanged. All rank/count/'
    'threshold metrics were exact locally. One NoGraph seed-404 recipe/scenario BCE value '
    'differed by -1.1102230246251565e-16 across the local and Kaggle environments; this '
    'difference is recorded without introducing a tolerance or changing any reported '
    'Kaggle value. Kaggle metric recomputation passed exact equality in the execution environment.')

per_seed = []
for kind, title in [('scenario_macro', 'Scenario-macro metrics'), ('pooled', 'Pooled metrics')]:
    rows = [([str(run_data[key]['seed']), display[run_data[key]['method']]], run_data[key][kind]) for key in order]
    per_seed.append(title + '\n\n' + table(rows, ['Seed', 'Method']))
selection = ['| Seed | Method | Selected epoch | Executed epochs | Minimum validation macro BCE | Corruption threshold |',
             '| --- | --- | --- | --- | --- | --- |']
for key in order:
    d = run_data[key]
    selection.append(f"| {d['seed']} | {display[d['method']]} | {d['selected_epoch']} | {d['epochs_executed']} | "
                     f"{number(d['minimum_validation_scenario_macro_BCE'])} | {number(d['threshold_selection']['threshold'])} |")
report = report.replace('# Primary EpistemicGAT vs StandardGAT Contrast',
    '\n\n'.join(per_seed) + '\n\nSelected checkpoints and frozen thresholds\n\n' + '\n'.join(selection) +
    '\n\n# Primary EpistemicGAT vs StandardGAT Contrast', 1)

rows = [([str(run_data[key]['seed']), display[run_data[key]['method']], scenario], values)
        for key in order for scenario, values in scenario_data[key].items()]
report = report.replace('# Recipe-Level Results', table(rows, ['Seed', 'Method', 'Scenario']) + '\n# Recipe-Level Results', 1)

recipe_tables = []
for kind, title in [('scenario_macro', 'Recipe scenario-macro metrics'), ('pooled', 'Recipe pooled metrics')]:
    rows = [([str(run_data[key]['seed']), display[run_data[key]['method']], recipe], values[kind])
            for key in order for recipe, values in recipe_data[key].items()]
    recipe_tables.append(title + '\n\n' + table(rows, ['Seed', 'Method', 'Recipe']))
report = report.replace('# Calibration Metrics', '\n\n'.join(recipe_tables) + '\n# Calibration Metrics', 1)
report = report.replace('preregistered t4 conditional seed interval=[np.float64(-5.4543476426879355e-05), np.float64(0.00010744540516811291)]',
    'preregistered t4 conditional seed interval=[-5.4543476426879355e-05, 0.00010744540516811291]')
target = ENGINEERING / 'delivery_report.md'
assert not target.exists(), 'Do not overwrite a delivery report'
target.write_text(report, encoding='utf-8')

completion = read(ENGINEERING / 'kaggle_completion_record.json')
completion['status'] = 'ALL_15_COMPLETE_SEALED_AND_LOCALLY_VERIFIED'
completion['download_verification'] = {
    'passed': True,
    'archive': str(ENGINEERING / 'kaggle_paired_results_download.zip'),
    'source_archive': r'C:\Users\Aditya\OneDrive\Desktop\carla_gat_paired_results_v1.zip',
    'archive_sha256': verification['archive_sha256'],
    'sealed_files_verified': verification['sealed_files_verified'],
    'checkpoint_internal_hashes_verified': verification['checkpoint_internal_hashes_verified'],
    'protected_files_unchanged': verification['preserved_workspace_files'],
    'verification_report': str(ENGINEERING / 'local_import_verification.json'),
    'local_results_root': verification['local_results_root'],
    'local_raw_root': verification['local_raw_root'],
    'delivery_report': str(target),
}
(ENGINEERING / 'kaggle_completion_record.json').write_text(json.dumps(completion, indent=2) + '\n', encoding='utf-8')
readme = ENGINEERING / 'README.md'
readme.write_text(readme.read_text(encoding='utf-8') + '\n\n## Completed local delivery\n\n'
    'All 15 real training runs completed without a failed training run. The ledger was '
    'sealed before aggregation. The optional attention audit initially stopped due to '
    'validation-only rebatching; the failure was retained and the audit recovered using '
    'original full-dataset batch boundaries, requiring bit-exact saved predictions. '
    'No scientific code, input, tolerance or training run was changed. The correct UTF-8 '
    'recovery source is `aggregate_results_batch_aligned_v2.py`, SHA-256 '
    '`0d65219c778fd5c8680c782e92a529c48627e40f8e4e930e3d9df21d5e02397e`. '
    'The earlier incorrectly decoded engineering generation was never executed and remains preserved.\n\n'
    'The user-downloaded archive was imported and verified: 669 sealed files, '
    '426 checkpoint internal hashes, all 15 predictions/metrics and frozen statistical '
    'recomputations; 1,691 protected files and upstream/graph identities remain unchanged. '
    'Raw runs are in `../carla_gat_paired_raw_runs_v1/`; sealed results are in '
    '`../carla_gat_paired_runs_v1/`. See `local_import_verification.json` and '
    '`delivery_report.md` for the completed 18-section report and readable frozen metric tables. '
    'The original Kaggle result seal was not modified. No TEST, CAL arrays, conformal, '
    'secondary GNSS, tuning, commit or push occurred.\n', encoding='utf-8')

files = ['delivery_report.md', 'local_import_verification.json', 'kaggle_completion_record.json',
         'README.md', 'finalize_delivery.py', 'import_results.py', 'aggregate_results_batch_aligned_v2.py']
seal = ENGINEERING / 'local_delivery_SHA256SUMS'
seal.write_text(''.join(f'{sha(ENGINEERING/name)}  {name}\n' for name in files), encoding='utf-8')
(ENGINEERING / 'local_delivery_SHA256SUMS.sha256').write_text(f'{sha(seal)}  {seal.name}\n', encoding='utf-8')
print(json.dumps({'passed': True, 'delivery_report': str(target), 'local_delivery_seal_sha256': sha(seal),
                  'original_results_seal_unchanged': sha(RESULTS/'SHA256SUMS') == verification['results_seal_sha256']}, indent=2))
