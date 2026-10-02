"""Final seventeen-section report: fixed N20 data, unchanged science."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=Path(__file__).parent
from history import histories
def read(name):return json.loads((OUT/name).read_text())
def f(x):return f'{x:.9g}'
def span(x):return '['+', '.join(f(v) for v in x)+']' if x is not None else 'none'
config=read('frozen_acquisition_config.json');acq=read('acquisition_result.json');manifest=read('frozen_dataset_manifest.json');cache=read('cache_result.json');cal=read('calibration_results.json');export=read('kaggle_artifact_result.json');integrity=read('integrity_results.json');pipeline=read('pipeline_result.json');reuse=read('compute_reuse_results.json')
agents=cal['agents'];past=histories(ROOT)
for name,r in agents.items():
 past[name]['20']={'n_normal':r['CAL_normal_rows'],'n_pseudo':r['CAL_pseudo_rows'],'audit':r['audit'],'valid':r['calibration_valid'],
  'uq':r['scientific_probability_and_uq'],'raw_member_score_disagreement':r['raw_score_disagreement'],
  'calibrated_probability_disagreement':r['calibrated_probability_disagreement'],
  'score_distributions':{label:{'ensemble_mean':r['ensemble_mean_distributions'][label],'pooled_members':r['member_score_distributions'][label]} for label in ['normal','pseudo']}}
lines=[]
def section(title):lines.extend(['# '+title,''])
def para(text):lines.extend([text,''])
def table(headers,rows):
 lines.append('| '+' | '.join(headers)+' |');lines.append('|'+'|'.join(['---']*len(headers))+'|')
 lines.extend('| '+' | '.join(str(v) for v in row)+' |' for row in rows);lines.append('')
section('Frozen N=20 Protocol')
para('N=20 is the final planned pre-GAT TRAIN-development expansion. Residual calibration invalidity leads to a methodology decision, with no automatic proposal or acquisition beyond twenty scenarios.')
para(f"Frozen before network access: acquisition config SHA-256 `{acq['config_sha256']}`; compute-reuse policy SHA-256 `{manifest['compute_reuse_policy_sha256']}`. First twenty complete official TRAIN scenarios in parsed archive order, whole scenarios, PCG64 seed 2026, exactly 15 FIT / 5 CAL, window 12, unchanged feature/normality/calibration/audit/recipe code and default severities. No performance-dependent selection.")
section('Network / Prefix Integrity')
para(f"N=10 prefix before/after: {acq['original_after']['bytes']:,} bytes, SHA-256 `{acq['original_after']['sha256']}`. A real independent N=20 staging copy was verified before appending. Original 3 GB prefix is also byte-hash verified unchanged. No N=10/N=2 data or artifacts were overwritten.")
para(f"One sequential N=10-prefix replay reconstructed the live gzip/TAR state because the previous process's zlib state could not safely be serialized. HTTP 206 Range resumed at byte {config['original_prefix']['bytes']:,}, with exact offset/length/encoding checks and stable validator {acq['remote_validator']!r}. No initial-prefix redownload or independent later-range decompression. Added network bytes: {acq['network_bytes']:,}; final N=20 prefix: {acq['staged_bytes']:,} bytes, SHA-256 `{acq['staged_sha256']}`.")
para(f"Stopped immediately after twenty complete scenarios at `{json.dumps(acq['trailing_boundary'])}`. At most one 256 KiB compressed chunk contains boundary read-ahead. The trailing scenario was not counted or extracted. Acquisition log: `{config['staging_root']}/acquisition_log.jsonl`.")
section('20 Complete TRAIN Scenarios')
table(['Ordinal','Scenario','Town','RGB / Seg count and range','GNSS / IMU rows','Reused raw','Partition'],[[r['archive_ordinal'],r['scenario_id'],r['town'],f"{r['tick_coverage']['rgb_count']}/{r['tick_coverage']['seg_count']}, {r['tick_coverage']['rgb_range']}",f"{r['tables']['gnss.feather']['rows']}/{r['tables']['imu.feather']['rows']}",r['reused'],r['partition']] for r in manifest['scenarios']])
para('Every scenario has complete consumed members, contiguous aligned RGB/Seg coverage, readable required Base feather tables, aligned sensor indexes and finite required GNSS/IMU values, strict TRAIN loader validation and proven subsequent-scenario completion evidence. The frozen dataset manifest records first TAR offsets, all feather names/schemas/counts, actual paths and per-source-file sizes/SHA-256 hashes.')
section('Town Diversity')
para(f"Unique towns: {manifest['unique_towns']}. Scenarios per town: `{json.dumps(manifest['town_counts'])}`. Selection was retained exactly in archive order.")
if manifest['unique_towns']==1:para('Limitation: scenario diversity without cross-town diversity.')
else:para('This checkpoint includes the recorded towns only; it makes no held-out town or final generalization claim.')
section('Frozen 15/5 FIT-CAL Manifest')
para(f"Dataset manifest SHA-256 `{(OUT/'frozen_dataset_manifest.sha256').read_text().strip()}`. Frozen after all twenty scenarios validated, before agent fitting. Sorted canonical IDs and existing PCG64 seed 2026 rule were applied once to the full list; N=10 membership was not manually extended.")
para('FIT_NORMAL: '+', '.join(manifest['FIT_NORMAL'])+'.')
para('CAL_NORMAL: '+', '.join(manifest['CAL_NORMAL'])+'.')
table(['Earlier scenario','N=10 role','N=20 role'],[[sid,r['old'],r['new']] for sid,r in manifest['N10_role_changes'].items()])
section('N=20 Compact Cache')
para(f"One production build/assembly: {cache['clean_rows']:,} clean rows ({cache['FIT_rows']:,} FIT / {cache['CAL_rows']:,} CAL), plus {cache['CAL_pseudo_rows']:,} CAL-derived pseudo rows. Schema {cache['cache_schema_version']}, content SHA-256 `{cache['content_sha256']}`. NPZ: {cache['npz_bytes']:,} bytes. Measured preprocessing/assembly: {cache['preprocessing_wall_s']/60:.2f} minutes.")
para(f"Reused {reuse['clean_reused_rows']:,} clean rows across all first ten scenarios exactly, after code/window/schema/source hash and partition-independence checks. Only the ten newly acquired scenarios underwent clean raw feature extraction. Production builder ordering, serialization, manifest construction and content hashing are unchanged; orchestration returns verified old feature blocks with N=20 partition labels.")
para(f"Reused CAL pseudo scenarios: {reuse['pseudo_reused_scenarios']} ({reuse['pseudo_reused_rows']:,} rows). Excluded prior CAL→FIT pseudo scenarios: {reuse['old_CAL_to_FIT_pseudo_excluded']}. Other N=20 CAL pseudo rows were generated by unchanged production recipes. Final old clean features are byte-identical; eligible pseudo rows equal their old arrays exactly. No CAL pseudo enters FIT graph training, and no graph rows were generated.")
para('Verified all scenario/tick/partition counts, finite feature blocks, schema/content hashes, CAL-only pseudo parentage, target_normal semantics, causal parent windows and exact untouched modality blocks. No second deterministic rebuild.')
section('Four-Agent Calibration Validity')
table(['Agent','FIT scenarios / rows','CAL scenarios / clean / pseudo','Optimizer converged','Gradient norm','Min Hessian eigenvalue','Pooled structure','Boundary trend','finite_optimum / valid'],[[name,f"{r['FIT_scenarios']}/{r['FIT_rows']}",f"{r['CAL_scenarios']}/{r['CAL_normal_rows']}/{r['CAL_pseudo_rows']}",r['audit']['optimizer_converged'],f(r['audit']['gradient_norm']),f(r['audit']['minimum_curvature']),r['audit']['separation']['pooled_members']['classification'],r['audit']['boundary_trend'],f"{r['audit']['finite_optimum']}/{r['calibration_valid']}"] for name,r in agents.items()])
table(['Agent','Mean normal / pseudo range','Pooled normal / pseudo range','Pooled overlap','Best mean threshold','Violations / observations','Numerical status'],[[name,span(r['audit']['separation']['ensemble_mean']['normal_range'])+' / '+span(r['audit']['separation']['ensemble_mean']['pseudo_range']),span(r['audit']['separation']['pooled_members']['normal_range'])+' / '+span(r['audit']['separation']['pooled_members']['pseudo_range']),span(r['audit']['separation']['pooled_members']['overlap_interval']),f(r['audit']['separation']['ensemble_mean']['best_threshold']),f"{r['audit']['separation']['ensemble_mean']['violations']}/{r['CAL_normal_rows']+r['CAL_pseudo_rows']}",r['audit']['status']] for name,r in agents.items()])
para('Thresholds are diagnostic only. Calibration objective, positive-slope parameterization, optimizer and validity audit are unchanged. Convergence alone does not establish validity; the report retains gradients, curvature, structural geometry, boundary/profile diagnostics and the complete audit in calibration_results.json. Accepted finite stationary candidates are numerical evidence, without a global-existence claim. Invalid scientific probability/UQ remains null.')
def geometry_table(names):
 rows=[]
 for name in names:
  for n,r in past[name].items():
   g=r['audit']['separation']['ensemble_mean'];m=r['audit']['separation']['pooled_members'];count=r['n_normal']+r['n_pseudo']
   rows.append([name,n,g['classification'],m['classification'],span(g['overlap_interval']),f"{g['violations']}/{count} ({100*g['violations']/count:.4f}%)",r['valid'],r['audit']['status']])
 table(['Agent','N','Mean geometry','Member geometry','Mean overlap','Violations','Valid','Status'],rows)
section('Camera N=2→10→20')
geometry_table(['Camera'])
camera=agents['Camera'];a=camera['audit'];g=a['separation']['ensemble_mean']
if camera['calibration_valid']:
 para(f"Camera reaches an accepted finite stationary calibration at N=20 under the unchanged protocol: {a['n_iter']} optimizer iterations, raw slope {f(a['slope_raw'])}, gradient {f(a['gradient_norm'])}, minimum curvature {f(a['minimum_curvature'])}, boundary trend {a['boundary_trend']}. This is development evidence on the frozen constructed clean-vs-pseudo task, with no final generalization claim.")
 para(f"Finite calibration validity does not establish strong corruption discrimination: normal/pseudo probability medians are {f(camera['scientific_probability_and_uq']['normal']['prob_normal']['median'])}/{f(camera['scientific_probability_and_uq']['pseudo']['prob_normal']['median'])}, with substantial overlapping score distributions and the diagnostic threshold violations reported above.")
elif g['classification']=='overlapping':para(f"Camera remains overlapping and scientifically unvalidated. Observed numerical status: {a['status']}; optimizer convergence={a['optimizer_converged']}, iterations={a['n_iter']}, gradient={f(a['gradient_norm'])}, minimum curvature={f(a['minimum_curvature'])}, boundary trend={a['boundary_trend']}, fitted raw slope={f(a['slope_raw'])}. This reports the finite-optimization/audit result as observed; overlap does not prove a finite optimum, and optimizer failure does not prove mathematical separation.")
else:para(f"A different Camera geometry appears: mean={g['classification']}, pooled members={a['separation']['pooled_members']['classification']}; unchanged audit status={a['status']}. No optimizer/calibrator change was made.")
section('GNSS N=2→10→20')
geometry_table(['GNSS'])
diag=agents['GNSS']['pre_exponential_diagnostics']
table(['CAL class','Member entries / exact zeros','All-zero observations','Squared distance range','Distance / scale range','Exp argument range','Smallest positive score'],[[label,f"{d['member_entries']}/{d['exact_zero_entries']} ({100*d['exact_zero_fraction']:.6g}% zero)",d['all_members_zero_observations'],span([d['squared_mahalanobis']['min'],d['squared_mahalanobis']['max']]),span([d['normalized_squared_distance']['min'],d['normalized_squared_distance']['max']]),span([d['exponential_argument']['min'],d['exponential_argument']['max']]),f(d['smallest_positive_observed_score']) if d['smallest_positive_observed_score'] is not None else 'none'] for label,d in diag['classes'].items()])
para(f"Unchanged score: exp(-0.5*d2/d2_scale). Diagnostic distances use the fitted members' existing mahalanobis_distance2 method; exact scalar score reproduction was verified. Float64 smallest positive subnormal={diag['float64_smallest_positive_subnormal']}, smallest positive normal={diag['float64_smallest_positive_normal']}, log(smallest subnormal)={f(diag['log_smallest_positive_subnormal'])}. Per-member scales: {diag['distance_scale_per_member']}. Full p10/median/p90 distance and exponent statistics are saved in gnss_distance_diagnostics.json.")
for label,d in diag['classes'].items():para(f"{label}: {d['arguments_below_log_smallest_subnormal']} exponent arguments fall below log(smallest positive subnormal); all recorded exact-zero entries explained by this underflow condition={d['all_zero_scores_explained_by_exp_underflow']}.")
if diag['classes']['pseudo']['exact_zero_entries']==diag['classes']['pseudo']['member_entries'] and diag['classes']['normal']['all_members_zero_observations']==0:
 para(f"Increasing TRAIN/CAL scenario diversity from N=2 → N=10 → N=20 did not remove the GNSS separation pathology. At N=20 every pseudo member score remains exactly zero, and every normal ensemble mean remains strictly positive. Pooled-member geometry changes from complete separation at N=2/N=10 to {agents['GNSS']['audit']['separation']['pooled_members']['classification']} at N=20: {diag['classes']['normal']['exact_zero_entries']} of {diag['classes']['normal']['member_entries']} normal member entries also underflow to zero. This is an endpoint tie, not substantial class overlap; no normal observation has all five member scores zero.")
 para('For these stored scores there is no finite unregularized ensemble-predictive calibration MLE: at any finite intercept, increasing the positive slope leaves every all-zero pseudo probability unchanged while strictly increasing every normal ensemble probability, since each normal observation has at least one positive member score. The mathematical likelihood therefore continues to improve with slope. Numerical convergence does not override the unchanged invalidity audit. Large Mahalanobis distances drive the quantified float64 underflow; no explicit score clamp was used. Recovering numerical score resolution alone would not guarantee removal of structural separation.')
else:para('GNSS geometry changed as shown above. Exact-zero frequency and distance diagnostics are reported independently of the structural classification; no score transform was changed.')
section('Seg / IMU Stability')
geometry_table(['Seg','IMU'])
def quantiles(d):return '/'.join(f(d[k]) if k in d else 'unavailable' for k in ['p10','median','p90','max'])
rows=[]
for name in ['Seg','IMU']:
 for n,r in past[name].items():
  if r['uq'] is None:rows.append([name,n,'both','unavailable']);continue
  for label,d in r['uq'].items():rows.append([name,n,label,quantiles(d['prob_normal'])])
table(['Agent','N','CAL class','prob_normal p10 / median / p90 / max'],rows)
para('N=2 accepted results retain historical medians only; unsaved probability quantiles are marked unavailable. N=10/N=20 have full saved distributions. These changes also reflect the recomputed scenario-level split and different CAL composition; the comparison is descriptive, without hyperparameter changes or an isolated causal claim.')
section('UQ N=20')
para('Exactly preserved: total=H(mean p_t), aleatoric=mean H(p_t), epistemic=max(0,total−aleatoric). Entropy uses the production natural-log convention. Only accepted valid mappings contribute probability-space UQ; invalid mappings remain unavailable.')
rows=[]
for name,r in agents.items():
 if not r['calibration_valid']:continue
 for label,d in r['scientific_probability_and_uq'].items():rows.append([name,label,quantiles(d['prob_normal']),quantiles(d['epistemic']),quantiles(d['total']),quantiles(d['aleatoric'])])
table(['Valid agent','CAL class','prob_normal p10 / median / p90 / max','Epistemic p10 / median / p90 / max','Total p10 / median / p90 / max','Aleatoric p10 / median / p90 / max'],rows)
rows=[]
for name,versions in past.items():
 for n,r in versions.items():
  if not r['valid']:continue
  for label,d in r['uq'].items():
   raw=r['raw_member_score_disagreement'][label];prob=r['calibrated_probability_disagreement'][label] if r['calibrated_probability_disagreement'] is not None else None
   rows.append([name,n,label,quantiles(d['epistemic']),f(d['total']['median']),f(d['aleatoric']['median']),quantiles(raw),quantiles(prob) if prob is not None else 'unavailable'])
table(['Agent','N','CAL class','Epistemic p10 / median / p90 / max','Total median','Aleatoric median','Raw score std p10 / median / p90 / max','Probability std p10 / median / p90 / max'],rows)
para((OUT/'uq_interpretation.txt').read_text().strip())
para('Historical N=2 medians come from accepted valid Seg/IMU outputs already saved in the accepted audit; their full probability-member distributions/intercept were not retained, so missing values are explicitly unavailable. N=10 probability disagreement is reconstructed read-only from saved accepted parameters/member scores, without refitting. Full historical comparison records are stored in comparison_results.json.')
section('N=20 Kaggle Handoff')
para(f"Prepared handoff v2: `{export['artifact_path']}`, {export['total_bytes']:,} bytes, {export['n_rows']:,} rows. No raw images/archives, graph examples or graph training. Includes frozen scenario/town/tick/15–5 partition metadata, compact features with explicit offsets, five bootstrap member scores per agent, validity/status, nullable calibrated prob_normal and canonical UQ, per-agent and observation constructed targets, pseudo parent/recipe/severity/seed/causal-window provenance, fitted parameters and cache/protocol/code/reuse hashes.")
para('Invalid prob_normal/total/aleatoric/epistemic are NaN with an explicit per-agent validity mask, meaning scientific null. Prediction semantics remain P(target=normal) under the TRAIN-derived constructed clean-vs-pseudo task, without a physical-safety claim. All pseudo rows are CAL-only and cannot be repurposed as FIT graph-training examples.')
section('Storage / Timing')
s=integrity['storage']
para(f"Acquisition including independent copy/replay/validation: {acq['elapsed_s']/60:.2f} minutes. N=20 cache preprocessing/assembly: {cache['preprocessing_wall_s']/60:.2f} minutes; analysis/export: {pipeline['agent_analysis_wall_s']/60:.2f} minutes. N20 staged prefix: {s['N20_prefix_bytes']/1e9:.3f} GB; newly extracted raw data: {s['new_raw_bytes']/1e9:.3f} GB; final cache total: {s['cache_bytes']/1e6:.3f} MB; handoff: {s['handoff_bytes']/1e6:.3f} MB. All older prefixes/raw/cache/export artifacts remain separately preserved.")
para('The measured N=20 preprocessing time includes clean extraction for only ten new scenarios plus newly needed CAL pseudo generation and deterministic assembly. It excludes the avoided repeat extraction of N=10 clean rows. Earlier N=10 preprocessing took 67.92 minutes; that is context rather than a measured counterfactual speedup.')
para(f"Verified new clean feature blocks retained for reuse: {s['new_clean_blocks_bytes']/1e6:.3f} MB. The staging inventory records these blocks, raw/source manifests, logical view, final cache and handoff locations; original data and prior checkpoints remain separate.")
para(f"New clean extraction overlapped acquisition: measured active clean-block intervals total {cache['new_clean_precompute_active_wall_s']/60:.2f} minutes; final assembly/CAL-pseudo wall interval is {cache['cache_assembly_and_CAL_pseudo_wall_s']/60:.2f} minutes. The reported preprocessing sum is these measured intervals, not an additional serial elapsed delay. Per-scenario clean blocks contain no FIT/CAL assignment, models or pseudo rows; all final labels were assigned from the frozen twenty-scenario manifest.")
section('Tests / Core Integrity')
for label,name in [('Focused','focused_tests.txt'),('Exact requested broad suite','broad_tests.txt')]:
 text=(OUT/name).read_text();matches=re.findall(r'=+\s*([^\n]*passed[^\n]*)\s*=+',text)
 assert matches and 'failed' not in matches[-1]
 para(label+': '+matches[-1].strip('= ')+'.')
para('Broad baseline remains 505 tests; no production source or test files were changed. Existing N=2/N=10 reports are byte-hash preserved, original 3 GB and N=10 12.548 GB prefixes match their accepted hashes, all prior raw/artifact sizes/mtimes and cache/export byte hashes match the initial snapshot, generic/frozen source/tests remain unchanged, bare import cognix loads no CARLA adapters, and tracked source/test diff is empty.')
para('No official TEST access, GAT/conformal changes, calibration/normality-score/feature/recipe/severity/threshold tuning, added regularization, slope caps, commits/pushes or destructive Git operations. Repository additions are confined to reports/carla_train_expand20_v1; the file inventory is in integrity_results.json. New temporary staging/raw/cache/export locations are explicitly recorded in the configuration and result manifests.')
section('Final Pre-GAT Modality Status')
statuses={}
for name,r in agents.items():
 a=r['audit'];geometry=a['separation']['pooled_members']['classification']
 if r['calibration_valid']:status='calibrated-valid'
 elif geometry in ['completely_separated','quasi_separated']:status='calibrated-invalid due structural separation ('+geometry+')'
 else:status='calibrated-invalid due finite-optimization/audit issue ('+a['status']+')'
 statuses[name]=status
table(['Modality','Final status'],list(statuses.items()))
para('This classification uses the frozen unchanged audit. It separates mathematical structural evidence from finite-optimization failure and keeps invalid probabilities/UQ unavailable. N=20 development data is frozen; no extra acquisition is proposed as a calibration remedy.')
section('Ready / Not Ready for GAT')
para('Not ready to start GAT in this milestone. The N=20 development checkpoint is available for a separately preregistered methodology decision. Any invalid probability channels require an explicit scientific handling decision; graph sampling/targets/evaluation and FIT pseudo generation are also a later authorized stage. No graph construction, StandardGAT/EpistemicGAT training or conformal fitting occurred.')
section('Recommended Methodology Decision')
invalid=[name for name,r in agents.items() if not r['calibration_valid']]
if invalid:
 para('Stop TRAIN expansion at N=20 and choose how to handle invalid mappings: '+', '.join(invalid)+'. Preserve this dataset and the negative results. Additional scenario acquisition is not the automatic next step.')
 para('For the next milestone only: retain raw/member normality scores as uncalibrated statistics, or exclude invalid calibrated probability/UQ channels under a preregistered downstream comparison. If changing calibration, preregister the model/objective, validity criteria and evaluation before fitting; a regularized model would be a scientific protocol change, not a repair of the current unregularized MLE.')
 if not agents['GNSS']['calibration_valid']:para('GNSS: distinguish loss of float64 exponential resolution from structural separation. A future score-representation study could retain pre-exponential distances/log-statistics, but recovering numeric resolution does not itself create a finite unregularized calibration MLE when the classes remain separated. A separately approved regularized/model alternative or exclusion decision would still need justification and validation.')
 if not agents['Camera']['calibration_valid']:para('Camera: preregister a finite-optimization/identifiability investigation of the observed overlapping score geometry and positive monotone calibration family. Any optimizer or calibration-family revision belongs to that future milestone; this result must remain invalid under the current rule.')
else:para('All four mappings have accepted finite stationary candidates. Keep N=20 frozen and preregister graph targets/sampling/evaluation before requesting graph work; these development results establish no final generalization claim.')
para('None of these next-milestone options was implemented. Stop here: no N>20 acquisition, official TEST, GAT, conformal, scientific tuning, regularization or commits/pushes.')
(OUT/'comparison_results.json').write_text(json.dumps(past,indent=2))
(OUT/'final_modality_status.json').write_text(json.dumps(statuses,indent=2))
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
assert len([line for line in lines if line.startswith('# ')])==17
integrity['new_milestone_files']=sorted(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file())
(OUT/'integrity_results.json').write_text(json.dumps(integrity,indent=2))
print('Saved complete seventeen-section N20 report')
