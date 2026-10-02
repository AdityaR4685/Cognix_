"""Verify downloaded results/checkpoints and local input preservation; never train."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/'reports/carla_gat_paired_runs_v1'
RAW=ROOT/'reports/carla_gat_paired_raw_runs_v1'

def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def require(c,m):
    if not c: raise RuntimeError(m)
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')
def check_seal(directory):
    require(digest(directory/'SHA256SUMS')==(directory/'SHA256SUMS.sha256').read_text().split()[0],'Seal checksum')
    count=0
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1); p=(directory/name).resolve()
        require(p.is_relative_to(directory.resolve()),'Unsafe seal path')
        require(digest(p)==expected,'File checksum '+str(p)); count+=1
    return count

def compare_metrics(local, sealed, path=''):
    """Record cross-platform loss/reliability rounding; require all ranks/counts.

The locked Kaggle analysis already requires exact metric recomputation there.
Local NumPy/OS differs; no new cross-environment acceptance tolerance is used.
Scientific reported numbers always remain the sealed Kaggle values.
"""
    differences=[]
    if isinstance(sealed,dict):
        require(isinstance(local,dict) and set(local)==set(sealed),'Metric structure '+path)
        for k in sealed: differences.extend(compare_metrics(local[k],sealed[k],path+'/'+k))
    elif isinstance(sealed,list):
        require(isinstance(local,list) and len(local)==len(sealed),'Metric list '+path)
        for i,(a,b) in enumerate(zip(local,sealed)): differences.extend(compare_metrics(a,b,path+'/'+str(i)))
    elif local!=sealed:
        leaf=path.rsplit('/',1)[-1]
        require(leaf in ('Brier','BCE','ECE','weighted_gap','mean_probability'), 'Discrete/rank/threshold metric mismatch '+path)
        differences.append({'path':path,'local':local,'sealed_Kaggle':sealed,'difference':local-sealed})
    return differences

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True); args=parser.parse_args()
    require(digest(args.archive)==args.archive_sha256,'Downloaded archive differs from observed Kaggle checksum')
    require(not DEST.exists() and not RAW.exists(),'Refuse existing results/raw outputs')
    DEST.mkdir(); RAW.mkdir()
    with zipfile.ZipFile(args.archive) as z:
        require(z.testzip() is None,'Archive CRC')
        for info in z.infolist():
            parts=Path(info.filename).parts
            require(len(parts)>1 and parts[0] in ('carla_gat_paired_runs_v1','cognix_full_paired_graph_v1_individual'),'Unexpected archive content')
            directory=DEST if parts[0]=='carla_gat_paired_runs_v1' else RAW
            target=directory.joinpath(*parts[1:]).resolve()
            require(target.is_relative_to(directory.resolve()),'Unsafe archive path')
            if info.is_dir(): target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as source,target.open('xb') as output: shutil.copyfileobj(source,output)
    sealed_count=check_seal(DEST)
    ledger=read(DEST/'run_ledger.json'); ledger_hash=digest(DEST/'run_ledger.json')
    require(ledger_hash==(DEST/'run_ledger.json.sha256').read_text().split()[0],'Ledger seal')
    require(ledger['complete'] and ledger['completed_runs']==15,'All 15 completed')
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import torch
    from cognix.adapters.carla import graph_training as tr, graph_training_data as gd, graph_training_metrics as gm
    data=gd.load_graph_dataset(ROOT/'reports/carla_kaggle_validation_bundle_v1/bundle/graph_artifact',ROOT/'reports/carla_gat_preregistration_v1')
    require([(r['method'],r['seed']) for r in ledger['runs']]==[(m,s) for s in gd.SEEDS for m in gd.METHODS],'Exact method/seed plan')
    validation_metrics=read(DEST/'per_seed_metrics.json')['runs']; primary={}; scenario_d=[]; runs={}; all_checkpoints=0
    for r in ledger['runs']:
        method,seed=r['method'],r['seed']; key=f'{method}_{seed}'; root=RAW/key; directory=root/method/f'seed_{seed}'
        sealed_count+=check_seal(root)
        summary=read(directory/'run_summary.json'); history=read(directory/'history.json')
        require(summary['method']==method and summary['seed']==seed and summary['status']=='complete','Run identity')
        require(digest(directory/'metrics.json')==r['metrics_sha256'] and digest(directory/'predictions.npz')==r['prediction_file_sha256']
                and digest(directory/'history.json')==r['history_sha256'] and digest(directory/'run_summary.json')==r['summary_sha256'],'Ledger run hashes')
        checkpoints=sorted(directory.glob('checkpoint_epoch_*.pt'))
        for ckpt in checkpoints:
            payload=torch.load(ckpt,map_location='cpu',weights_only=False)
            internal=payload.pop('checkpoint_content_sha256')
            require(tr.recursive_content_hash(payload)==internal,'Checkpoint internal checksum '+key+' '+ckpt.name)
            all_checkpoints+=1
        selected=directory/Path(summary['selected_checkpoint']).name
        model,opt,payload=tr.load_checkpoint(selected,data,'cpu')
        require(payload['checkpoint_content_sha256']==r['checkpoint_content_sha256'] and digest(selected)==r['checkpoint_file_sha256'],'Selected checkpoint')
        minimum=min(history,key=lambda h:h['validation_scenario_macro_BCE'])
        require(minimum['epoch']==summary['selected_epoch'] and minimum['validation_scenario_macro_BCE']==summary['best_validation_scenario_macro_BCE'],'Earliest strict minimum')
        with np.load(directory/'predictions.npz',allow_pickle=False) as z: predictions=dict(z)
        require(tr.recursive_content_hash(predictions)==r['prediction_content_sha256'],'Prediction scientific hash')
        require(np.array_equal(predictions['row_key'],data.arrays['row_key']) and
                np.array_equal(predictions['scenario'],data.scenarios) and
                np.array_equal(predictions['split'],data.arrays['split']) and
                np.array_equal(predictions['target_normal'],data.arrays['target_normal']) and
                np.array_equal(predictions['p_corrupt'],1-predictions['q_normal']),'Prediction row identity')
        require(np.all(predictions['method']==method) and np.all(predictions['seed']==seed) and
                np.all(predictions['epoch']==summary['selected_epoch']) and
                np.all(predictions['checkpoint_content_sha256']==r['checkpoint_content_sha256']),'Prediction run metadata')
        recomputed=gm.dataset_metric_report(data,predictions['q_normal'])
        local_numeric_differences=compare_metrics(recomputed,read(directory/'metrics.json'))
        compare_metrics(recomputed['per_split']['GRAPH_VALIDATION']['scenario_macro'],validation_metrics[key]['scenario_macro'])
        evidence=DEST/'observer'/key
        require(digest(evidence/'paired_integrity.json')==r['paired_integrity_sha256'],'Pairing evidence seal')
        pairing=read(evidence/'paired_integrity.json'); completed=read(evidence/'observer_completion.json')
        require(pairing['passed'] and pairing['verified_before_optimization'] and completed['passed'],'Actual observer gate')
        for h in history:
            expected=pairing['epoch_plan'][h['epoch']-1]
            require(expected['row_order_sha256']==h['train_row_order_sha256'] and expected['batch_sizes']==h['batch_sizes'],'Executed epoch batching')
        final=torch.load(evidence/'observer_final_state.pt',map_location='cpu',weights_only=False)
        require(tr.recursive_content_hash(final)==read(evidence/'final_state.json')['content_sha256'],'Final executed state checksum')
        runs[key]={'selected_epoch':summary['selected_epoch'],'epochs_executed':summary['epochs_executed'],
                   'preserved_minimum_checkpoints':len(checkpoints),'all_internal_checkpoint_hashes_verified':True,
                   'prediction_and_metric_hashes_verified':True,'actual_paired_evidence_verified':True}
        runs[key]['local_cross_environment_metric_comparison']={'rank_count_threshold_metrics_exact':True,
            'all_metrics_bit_exact':not local_numeric_differences,'numeric_differences':local_numeric_differences,
            'new_tolerance':None,'reported_scientific_values':'unchanged sealed Kaggle results',
            'actual_Kaggle_recomputation':'exact equality required and passed by sealed aggregator'}
        del model,opt,payload,final
    scenarios=sorted(data.splits['GRAPH_VALIDATION']); values=read(DEST/'scenario_metrics.json')['runs']
    for seed in gd.SEEDS:
        a=read(DEST/'observer'/f'standard_gat_{seed}'/'paired_integrity.json')
        b=read(DEST/'observer'/f'epistemic_gat_{seed}'/'paired_integrity.json')
        require(a['actual_initial_content_sha256']==b['actual_initial_content_sha256'] and a['dropout_rng_sha256']==b['dropout_rng_sha256']
                and a['epoch_plan']==b['epoch_plan'] and a['optimizer_settings']==b['optimizer_settings'],'Actual paired seed matching')
        primary[seed]=validation_metrics[f'epistemic_gat_{seed}']['scenario_macro']['AUROC']-validation_metrics[f'standard_gat_{seed}']['scenario_macro']['AUROC']
        scenario_d.append([values[f'epistemic_gat_{seed}'][s]['AUROC']-values[f'standard_gat_{seed}'][s]['AUROC'] for s in scenarios])
    statistics=read(DEST/'statistical_summary.json')
    require(gm.paired_difference_summary(primary)==statistics['primary'],'Independent frozen paired/sign-flip result')
    require(gm.scenario_bootstrap(scenario_d)==statistics['whole_scenario_bootstrap'],'Independent frozen scenario bootstrap')
    snapshot=read(HERE/'preservation_snapshot.json')
    for name,expected in snapshot['workspace_files'].items(): require(digest(ROOT/name)==expected,'Preexisting workspace changed '+name)
    for group in ('upstream','artifact'):
        for name,expected in snapshot[group].items():
            p=ROOT/'reports/carla_gat_preregistration_v1/protocol.json' if name=='protocol.json' else Path(name)
            if not p.is_absolute(): p=ROOT/p
            require(digest(p)==expected,'Upstream/artifact changed '+name)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==snapshot['git_head'],'Git HEAD changed')
    require(digest(ROOT/'reports/carla_kaggle_environment_lock_v1/environment_lock.json')=='0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d','Local sealed environment lock')
    verification={'passed':True,'archive_sha256':args.archive_sha256,'run_ledger_sha256':ledger_hash,
        'results_seal_sha256':digest(DEST/'SHA256SUMS'),'sealed_files_verified':sealed_count,
        'checkpoint_internal_hashes_verified':all_checkpoints,'all_15_runs':runs,
        'preserved_workspace_files':len(snapshot['workspace_files']),'changed_preexisting_files':[],
        'upstream_and_graph_preserved':True,'git_HEAD_preserved':True,'no_commit_push':True,
        'frozen_statistical_recomputation_verified':True,'local_raw_root':str(RAW),'local_results_root':str(DEST)}
    write(HERE/'local_import_verification.json',verification)
    (HERE/'local_import_verification.json.sha256').write_text(digest(HERE/'local_import_verification.json')+'  local_import_verification.json\n')
    shutil.copyfile(args.archive,HERE/'kaggle_paired_results_download.zip')
    print(json.dumps(verification,indent=2))

if __name__=='__main__': main()
