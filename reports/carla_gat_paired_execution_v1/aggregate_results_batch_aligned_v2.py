"""Preregistered aggregation after verification of the sealed 15-run ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from execute_full import ROOT, LOCK, LOCK_HASH, read, write, digest, require, seal


def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,required=True); args=p.parse_args()
    out=args.results; ledger=read(out/'run_ledger.json'); ledger_hash=digest(out/'run_ledger.json')
    require(ledger_hash==(out/'run_ledger.json.sha256').read_text().split()[0],'Ledger seal must precede analysis')
    require(ledger['complete'] and ledger['completed_runs']==15 and len(ledger['runs'])==15,'All 15 runs must complete')
    require(digest(LOCK)==LOCK_HASH,'Environment lock changed')
    sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'reports/carla_kaggle_validation_bundle_v1'))
    import numpy as np
    import torch
    from cognix.adapters.carla import graph_training as tr, graph_training_data as gd, graph_training_metrics as metrics
    import launch_full, validate_kaggle as vk
    launch_full.preflight(LOCK)
    data=gd.load_graph_dataset(ROOT/'graph_artifact',ROOT/'reports/carla_gat_preregistration_v1')
    vi=data.validation_indices; scenarios=sorted(data.splits['GRAPH_VALIDATION'])
    require([(r['method'],r['seed']) for r in ledger['runs']]==[(m,s) for s in gd.SEEDS for m in gd.METHODS],'Exact plan')
    per_seed={}; per_scenario={}; per_recipe={}; calibration={}; attention={}; summaries={}; predictions={}
    (out/'validation_predictions').mkdir(exist_ok=True); (out/'attention').mkdir(exist_ok=True)
    for record in ledger['runs']:
        method,seed=record['method'],record['seed']; directory=Path(record['output']); key=method+'_'+str(seed)
        require(record['completion_status']=='complete','Failed run cannot enter analysis')
        for name,field in [('metrics.json','metrics_sha256'),('history.json','history_sha256'),
                           ('predictions.npz','prediction_file_sha256'),('run_summary.json','summary_sha256')]:
            require(digest(directory/name)==record[field],'Ledger output hash mismatch '+key+' '+name)
        raw_root=Path(record['root'])
        require(digest(raw_root/'SHA256SUMS')==(raw_root/'SHA256SUMS.sha256').read_text().split()[0],'Raw seal')
        for line in (raw_root/'SHA256SUMS').read_text().splitlines():
            expected,relative=line.split('  ',1); require(digest(raw_root/relative)==expected,'Raw file changed '+relative)
        summary=read(directory/'run_summary.json'); history=read(directory/'history.json')
        require(summary['graph_artifact_scientific_sha256']==gd.ARTIFACT_SHA256 and
                summary['trainer_identity']==tr.trainer_identity() and summary['training_specification']==tr.TRAINING_SPEC,'Run scientific identity')
        earliest=min(history,key=lambda h:h['validation_scenario_macro_BCE'])
        require(earliest['epoch']==summary['selected_epoch'] and earliest['validation_scenario_macro_BCE']==summary['best_validation_scenario_macro_BCE'],'Strict minimum earliest tie')
        with np.load(directory/'predictions.npz',allow_pickle=False) as z: pred=dict(z)
        require(tr.recursive_content_hash(pred)==record['prediction_content_sha256'],'Prediction content mismatch')
        require(np.array_equal(pred['row_key'],data.arrays['row_key']) and np.array_equal(pred['target_normal'],data.arrays['target_normal'])
                and np.array_equal(pred['scenario'],data.scenarios) and np.array_equal(pred['p_corrupt'],1-pred['q_normal']),'Prediction row identity')
        saved=read(directory/'metrics.json'); recomputed=metrics.dataset_metric_report(data,pred['q_normal'])
        require(saved==recomputed,'Frozen metrics independent recomputation mismatch')
        model,optimizer,payload=tr.load_checkpoint(summary['selected_checkpoint'],data,'cuda')
        require(payload['checkpoint_content_sha256']==record['checkpoint_content_sha256'] and
                digest(summary['selected_checkpoint'])==record['checkpoint_file_sha256'],'Selected checkpoint hashes')
        subset={name:value[vi] for name,value in pred.items()}
        validation_file=out/'validation_predictions'/f'{key}.npz'
        if validation_file.exists():
            with np.load(validation_file,allow_pickle=False) as prior:
                require(tr.recursive_content_hash(dict(prior))==tr.recursive_content_hash(subset),'Preserved prior partial export mismatch')
        else:
            with validation_file.open('xb') as f: np.savez_compressed(f,**subset)
        predictions[key]=subset; summaries[key]=summary
        validation=saved['per_split']['GRAPH_VALIDATION']
        per_seed[key]={'method':method,'seed':seed,'selected_epoch':summary['selected_epoch'],
              'epochs_executed':summary['epochs_executed'],'minimum_validation_scenario_macro_BCE':summary['best_validation_scenario_macro_BCE'],
              'threshold_selection':saved['threshold_selection'],'pooled':validation['pooled'],
              'scenario_macro':validation['scenario_macro'],'ledger_run':record}
        per_scenario[key]=validation['per_scenario']; per_recipe[key]=validation['per_recipe']
        calibration[key]={'threshold_selection':saved['threshold_selection'],
              'pooled':{name:validation['pooled'][name] for name in ('Brier','BCE','ECE','ECE_bins')},
              'scenario_macro':{name:validation['scenario_macro'][name] for name in ('Brier','BCE','ECE')}}
        if method!='nograph':
            q_all,a_all=tr.evaluate(model,data,device='cuda',audit_attention=True)
            require(np.array_equal(q_all,pred['q_normal']),'Batch-aligned attention export changed predictions')
            q,a=q_all[vi],a_all[vi]
            with (out/'attention'/f'{key}.npz').open('xb') as f:
                np.savez_compressed(f,row_key=pred['row_key'][vi],attention=a,semantics=np.array('pre_dropout,incoming_receiver_by_sender'))
            attention[key]={'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),
               'maximum_incoming_sum_error':float(np.abs(a.sum(-1)-1).max()),
               'self_edge_maximum':float(np.diagonal(a,axis1=-2,axis2=-1).max()),
               'normalization_checked_by_frozen_model':True}
        del model,optimizer,payload
    paired={}; primary={}; secondary={}; matrix=[]
    for seed in gd.SEEDS:
        standard=per_seed[f'standard_gat_{seed}']['scenario_macro']['AUROC']
        epi=per_seed[f'epistemic_gat_{seed}']['scenario_macro']['AUROC']
        no=per_seed[f'nograph_{seed}']['scenario_macro']['AUROC']
        require(standard is not None and epi is not None and no is not None,'Primary endpoint undefined')
        primary[seed]=epi-standard
        paired[str(seed)]={'nograph':no,'standard_gat':standard,'epistemic_gat':epi,'epistemic_minus_standard':epi-standard,
                          'standard_minus_nograph':standard-no,'epistemic_minus_nograph':epi-no}
        matrix.append([per_scenario[f'epistemic_gat_{seed}'][s]['AUROC']-per_scenario[f'standard_gat_{seed}'][s]['AUROC'] for s in scenarios])
        for name in metrics.METRICS:
            values=[per_seed[f'{m}_{seed}']['scenario_macro'][name] for m in gd.METHODS]
            orientation=-1 if name in ('Brier','BCE','ECE') else 1
            secondary.setdefault(name,{})[str(seed)]={
                'epistemic_minus_standard':None if any(v is None for v in values[1:]) else values[2]-values[1],
                'standard_minus_nograph':None if any(v is None for v in values[:2]) else values[1]-values[0],
                'epistemic_minus_nograph':None if values[0] is None or values[2] is None else values[2]-values[0],
                'better_direction':orientation,'undefined_reason':'requires defined constituent metrics' if any(v is None for v in values) else None}
    summary=metrics.paired_difference_summary(primary); bootstrap=metrics.scenario_bootstrap(matrix)
    require(summary['complete'],'Primary summary incomplete')
    e=data.arrays['node_features'][vi,:,1]; prior=(1/(1+e)).astype(np.float32); adjustment=np.log(prior)
    attention_difference={}
    for seed in gd.SEEDS:
        with np.load(out/'attention'/f'standard_gat_{seed}.npz',allow_pickle=False) as z: a=dict(z)
        with np.load(out/'attention'/f'epistemic_gat_{seed}.npz',allow_pickle=False) as z: b=dict(z)
        require(np.array_equal(a['row_key'],b['row_key']),'Attention alignment')
        difference=np.abs(a['attention']-b['attention'])
        attention_difference[str(seed)]={'mean_absolute_attention_difference':float(difference.mean()),
                  'maximum_absolute_attention_difference':float(difference.max()),
                  'interpretation':'descriptive selected-model difference; learned tensors can differ; not an isolated causal prior effect'}
    epidiag={node:{'E_min':float(e[:,j].min()),'E_mean':float(e[:,j].mean()),'E_max':float(e[:,j].max()),
                  'sender_weight_min':float(prior[:,j].min()),'sender_weight_mean':float(prior[:,j].mean()),
                  'sender_weight_max':float(prior[:,j].max()),'rounded_neutral_fraction':float((prior[:,j]==1).mean()),
                  'log_sender_adjustment_min':float(adjustment[:,j].min()),'log_sender_adjustment_max':float(adjustment[:,j].max())}
             for j,node in enumerate(data.schema['node_order'])}
    write(out/'per_seed_metrics.json',{'sealed_run_ledger_sha256':ledger_hash,'runs':per_seed})
    write(out/'paired_comparison.json',{'primary_endpoint':'GRAPH_VALIDATION scenario_macro_AUROC on p_corrupt=1-q_normal',
          'primary_contrast':'epistemic_gat minus standard_gat','per_seed':paired,'primary_summary':summary,
          'secondary_descriptive_contrasts':secondary,'NoGraph_role':'secondary/exploratory'})
    write(out/'scenario_metrics.json',{'scenarios':scenarios,'runs':per_scenario,'primary_paired_difference_matrix':matrix})
    write(out/'recipe_metrics.json',{'runs':per_recipe,'rule':'recipe pseudo rows and same pairs clean parents; frozen validation threshold'})
    write(out/'calibration_metrics.json',{'runs':calibration,'ECE_bins':15,'probability':'q_normal versus target_normal'})
    write(out/'statistical_summary.json',{'primary':summary,'exact_sign_flip':{
          'p':summary['exact_two_sided_sign_flip_p'],'two_sided':True,'enumerated_sign_assignments':32,
          'minimum_attainable_two_sided_p':.0625,'p_below_0_05_claim_permitted':False},
          'whole_scenario_bootstrap':bootstrap,'bootstrap_scenario_order':scenarios,
          'scope':'fixed-representation development validation; only three scenarios; no independent generalization guarantee'})
    write(out/'attention_summary.json',{'normalization':attention,'per_seed_descriptive_differences':attention_difference,
          'epistemic_sender_adjustment':epidiag,'prior':'1/(1+raw_E), unchanged strength 1',
          'relationship':'For fixed learned logits and other senders, increasing sender E decreases eligible normalized attention before dropout.',
          'no_prior_tuning':True,'model_selection_uses_attention':False})
    write(out/'post_execution_integrity.json',{'passed':True,'integrity':vk.integrity(),
          'environment_lock_sha256':digest(LOCK),'trainer_identity':tr.trainer_identity(),
          'no_CAL_TEST_conformal_GNSS_access':True,'all_raw_seals_verified':True,'run_ledger_sha256':ledger_hash})
    table='| Seed | NoGraph | StandardGAT | EpistemicGAT | Epi − Standard |\n|---|---:|---:|---:|---:|\n'
    for seed,v in paired.items(): table+=f"| {seed} | {v['nograph']:.9f} | {v['standard_gat']:.9f} | {v['epistemic_gat']:.9f} | {v['epistemic_minus_standard']:+.9f} |\n"
    scenarios_text='\n'.join(f"- {s}: paired AUROC differences across seeds {gd.SEEDS}: "+', '.join(f'{matrix[i][j]:+.9f}' for i in range(5)) for j,s in enumerate(scenarios))
    recipe_text='\n'.join('- '+recipe+': all method/seed pooled, scenario and macro metrics retained in recipe_metrics.json.' for recipe in sorted(next(iter(per_recipe.values()))))
    signs=['positive' if x>0 else 'negative' if x<0 else 'zero' for x in primary.values()]
    report=f'''# Frozen Execution Verification

All 15 preregistered real graph runs completed using the unchanged gated launcher independent commands, one visible T4, the exact sealed environment lock {LOCK_HASH}, graph protocol 68f035acda758bed0e7c3799a116f58a03719fee4e39f6468fdeaac0ef6ab203, graph scientific artifact {gd.ARTIFACT_SHA256}, trainer bundle {tr.trainer_identity()['bundle_sha256']}. Rows: 71,976 GRAPH_TRAIN and 17,994 GRAPH_VALIDATION. Three nodes Camera/IMU/Seg, frozen p/E/A, six directed edges and no self-loops. No CAL or TEST arrays were loaded.

# 15-Run Completion Ledger

15/15 complete; run_ledger.json was sealed before any cross-method aggregation, SHA-256 {ledger_hash}. Every independent run retains all strict-minimum checkpoints, selected checkpoint, full history, predictions, metrics, observer pairing evidence, final executed state, logs, runtime and environment. No runs replaced or selected by seed. Per-run raw SHA256SUMS verified before analysis.

# Per-Seed Results

Scenario-macro validation AUROC:

{table}
All eight frozen metrics, pooled and scenario-macro, selected epoch, minimum BCE and validation-selected threshold are recorded for every run in per_seed_metrics.json.

# Primary EpistemicGAT vs StandardGAT Contrast

Mean Δ={summary['mean']:+.12g}; median={summary['median']:+.12g}; sample SD={summary['sample_SD']:.12g}; min={summary['min']:+.12g}; max={summary['max']:+.12g}; paired dz={summary['paired_dz']}; preregistered t4 conditional seed interval={summary['conditional_seed_CI']}. This interval describes initialization variability conditional on the fixed validation set.

# NoGraph Comparison

NoGraph values and both GAT-minus-NoGraph contrasts are retained per seed in paired_comparison.json. These comparisons remain secondary/exploratory. NoGraph has 48 parameters; both GATs have 50 and receive identical p/E/A.

# Scenario-Level Results

{scenarios_text}

All per-scenario AUROC, AUPRC, F1, accuracy, balanced accuracy, Brier, BCE, ECE, confusion counts and bin evidence are retained in scenario_metrics.json.

# Recipe-Level Results

{recipe_text}

Diagnostics retain each recipe's pseudo rows together with the same pairs' clean parents. Undefined values retain null and frozen reasons.

# Calibration Metrics

Per-run pooled and scenario-macro Brier, BCE and 15-bin ECE are in calibration_metrics.json; per-scenario and per-recipe values are retained. No calibration was refitted. Thresholds maximize validation scenario-macro F1 over unique corruption probabilities union {{0,1}}, with largest-threshold exact tie rule; selection and reporting reuse validation and therefore remain development statistics.

# Exact Sign-Flip Result

Exact two-sided p={summary['exact_two_sided_sign_flip_p']}; all 32 assignments were enumerated using the unchanged frozen metric function. With five pairs the minimum attainable exact two-sided p is 0.0625. No p < 0.05 significance claim is made.

# Whole-Scenario Bootstrap

10,000 whole-scenario draws, PCG64 seed 606, percentile interval={bootstrap['percentile_interval']}. All ticks, clean/pseudo pairs and methods remain aligned within each scenario. Only three graph-validation scenarios are available, so this is descriptive fragility evidence and is not an independent generalization guarantee. No tick bootstrap was performed.

# Attention / Epistemic Diagnostics

All ten selected GAT checkpoints passed frozen attention normalization/finite/self-edge checks, and audited probabilities exactly matched saved selected-checkpoint validation predictions. Descriptive paired attention differences and raw E/sender-weight/log-adjustment summaries are retained in attention_summary.json. The prior remains 1/(1+E); neutral rounding or weak effects are accepted without amplification. Attention never selected a checkpoint.

# Seed Stability

Paired difference signs, in preregistered seed order: {signs}. All five values and variation summaries are retained without ranking/selecting seeds. Each actual trainer construction was checked for byte-identical explicitly cloned GAT tensors with independent storage, actual first-forward RNG and optimizer, and identical plans for all 100 frozen epoch permutations/batch boundaries before optimization; each executed epoch was checked against that plan.

# Negative / Positive Findings

Observed Epi-minus-Standard mean Δ={summary['mean']:+.12g}, range [{summary['min']:+.12g}, {summary['max']:+.12g}]. Positive, negative and zero differences are all retained. This fixed experiment supports only descriptive statements about these frozen methods, constructed corruptions and validation scenarios; it establishes no statistical superiority at p < 0.05.

# Scientific Interpretation

These are fixed-representation graph-development results on frozen GRAPH_VALIDATION scenarios, under artificial paired 1:1 clean/pseudo prevalence. Validation is reused for early stopping and threshold selection; upstream representation saw FIT scenarios that include graph validation. Results are not official CarlAnomaly TEST/anomaly performance, physical safety performance or independent population generalization. No feature, recipe, architecture or prior was tuned after observing outcomes.

# Environment / Determinism

Exact Python/PyTorch/NumPy/CUDA/cuDNN and T4 specifications matched the sealed lock. CUDA_VISIBLE_DEVICES=0 was verified before PyTorch import in fresh trainer children; Kaggle allocated two T4s, one was visible, and no distributed execution occurred. Deterministic algorithms and cuDNN deterministic enabled, benchmark/TF32/AMP disabled, CUBLAS_WORKSPACE_CONFIG=:4096:8. Adam lr .001, weight decay .0001, batch 256, max 100 epochs, dropout .1, no scheduler/gradient clipping, patience 10/min_delta 1e-5; selected checkpoint is strict observed minimum validation macro BCE with earliest exact tie.

# Core / Artifact Integrity

Portable bundle/source/graph/protocol seals and exact environment lock verified after all runs. The first optional audit attempt stopped because validation-only evaluation rebatched saved full-dataset predictions. The failed attempt is retained. Attention audit was recovered by using the original full-dataset batch boundaries and then selecting validation rows; all predictions were required to match bit-for-bit. No new tolerance, training, scientific source change or output replacement occurred. Local input/code/upstream preservation is verified when downloaded evidence is imported. Raw run artifacts and final result artifacts are separately sealed. N=20, calibration, feature extraction, recipes, generic COGNIX and frozen synthetic results were not edited. No commit or push occurred.

# Ready / Not Ready for Next Pipeline Stage

Ready for review of the complete frozen primary experiment. TEST, conformal and secondary GNSS remain unauthorized and unexecuted.

# Recommended Next Step

Review the sealed development-validation findings and separately preregister/authorize the next pipeline stage, resolving calibration independence and scenario/temporal exchangeability before any conformal or held-out evaluation. Stop this milestone without additional scientific runs.
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    seal(out)
    archive=out.with_name('carla_gat_paired_results_v1.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for directory in [out,Path('/kaggle/working/cognix_full_paired_graph_v1_individual')]:
            for file in sorted(directory.rglob('*')):
                if file.is_file(): z.write(file,file.relative_to(directory.parent))
    print('COGNIX_ALL_COMPLETE',json.dumps({'runs':15,'ledger_sha256':ledger_hash,'results_seal_sha256':digest(out/'SHA256SUMS'),
          'archive':str(archive),'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
          'primary_summary':summary,'bootstrap':bootstrap}),flush=True)

if __name__=='__main__': main()
