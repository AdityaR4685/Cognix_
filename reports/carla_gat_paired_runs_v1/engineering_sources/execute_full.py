"""Execute prepared independent commands, stop on any failure, seal before analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/kaggle/working/cognix_validation')
LOCK = Path('/kaggle/working/carla_kaggle_environment_lock_v1/environment_lock.json')
LOCK_HASH = '0e92846f45cb6d75e787c5d25b06f13bf1941b85104f8d6adf68e6973cd5de4d'
LAUNCH = ROOT/'reports/carla_kaggle_validation_bundle_v1/launch_full.py'
RAW = Path('/kaggle/working/cognix_full_paired_graph_v1_individual')
RESULT = Path('/kaggle/working/carla_gat_paired_runs_v1')

def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n')

def require(condition,message):
    if not condition: raise RuntimeError(message)

def seal(directory, name='SHA256SUMS'):
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','SHA256SUMS.sha256'))
    target = directory/name
    with target.open('x',encoding='utf-8') as f:
        f.writelines(digest(p)+'  '+p.relative_to(directory).as_posix()+'\n' for p in files)
    target.with_name(name+'.sha256').write_text(digest(target)+'  '+name+'\n')
    return digest(target)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--execute-authorized-full',action='store_true')
    args=parser.parse_args(); require(args.execute_authorized_full,'Explicit full execution required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Pre-import settings')
    require(digest(LOCK)==LOCK_HASH,'Exact sealed environment lock identity')
    require(not RAW.exists() and not RESULT.exists(),'Refuse existing experiment outputs')
    RESULT.mkdir()
    sys.path.insert(0,str(LAUNCH.parent)); sys.path.insert(0,str(ROOT))
    import launch_full
    import validate_kaggle as vk
    lock=launch_full.preflight(LOCK)
    write(RESULT/'environment_verification.json', {'passed':True,'exact_environment_lock_sha256':digest(LOCK),
          'environment':vk.actual_environment('cuda'),'integrity':vk.integrity(),
          'CUDA_VISIBLE_DEVICES_verified_before_torch_import':True})
    plan=[{'method':method,'seed':seed,'root':str(RAW/(method+'_'+str(seed)))}
          for seed in (101,202,303,404,505) for method in ('nograph','standard_gat','epistemic_gat')]
    write(RESULT/'execution_plan.json',{'runs':plan,'source':'sealed future_full_commands.md independent alternative',
          'no_replacement':True,'stop_on_any_failure':True,'scientific_functions_modified':False})
    env=os.environ.copy(); env['PYTHONPATH']=str(HERE)+os.pathsep+str(ROOT)
    env['COGNIX_FULL_OBSERVER']='1'; env['COGNIX_OBSERVER_ROOT']=str(RESULT/'observer')
    (RESULT/'logs').mkdir(); ledger=[]
    for item in plan:
        method,seed=item['method'],item['seed']; runroot=Path(item['root']); output=runroot/method/f'seed_{seed}'
        command=[sys.executable,str(LAUNCH),'--environment-lock',str(LOCK),'--output',str(runroot),
                 '--method',method,'--seed',str(seed),'--execute-full-experiment']
        started=time.time(); print('RUN_START',method,seed,flush=True)
        with (RESULT/'logs'/f'{method}_{seed}.log').open('xb') as f:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record={**item,'output':str(output),'returncode':process.returncode,'elapsed_seconds':time.time()-started}
        if process.returncode!=0:
            record['completion_status']='failed'; ledger.append(record)
            write(RESULT/'incomplete_run_ledger.json',{'complete':False,'runs':ledger,'remaining_runs':plan[len(ledger):],
                  'reason':'Stop on first failure; preserved logs, outputs, observer state; no replacement or retry'})
            raise RuntimeError('Frozen run failed: '+method+' '+str(seed))
        summary=read(output/'run_summary.json'); pairing=read(RESULT/'observer'/f'{method}_{seed}'/'paired_integrity.json')
        observed=read(RESULT/'observer'/f'{method}_{seed}'/'observer_completion.json')
        require(summary['status']=='complete' and pairing['passed'] and observed['passed'] and observed['first_forward_checked'],
                'Completion/pairing evidence missing')
        record.update({'completion_status':'complete','selected_epoch':summary['selected_epoch'],
              'epochs_executed':summary['epochs_executed'],
              'checkpoint_content_sha256':summary['checkpoint_content_sha256'],
              'checkpoint_file_sha256':digest(summary['selected_checkpoint']),
              'prediction_content_sha256':summary['prediction_content_sha256'],
              'prediction_file_sha256':digest(output/'predictions.npz'),'metrics_sha256':digest(output/'metrics.json'),
              'summary_sha256':digest(output/'run_summary.json'),'history_sha256':digest(output/'history.json'),
              'paired_integrity_sha256':digest(RESULT/'observer'/f'{method}_{seed}'/'paired_integrity.json')})
        seal(runroot); ledger.append(record); print('RUN_COMPLETE',method,seed,'epochs',summary['epochs_executed'],flush=True)
    for seed in (101,202,303,404,505):
        a=read(RESULT/'observer'/f'standard_gat_{seed}'/'paired_integrity.json')
        b=read(RESULT/'observer'/f'epistemic_gat_{seed}'/'paired_integrity.json')
        require(a['actual_initial_content_sha256']==b['actual_initial_content_sha256'] and
                a['dropout_rng_sha256']==b['dropout_rng_sha256'] and a['epoch_plan']==b['epoch_plan'] and
                a['optimizer_settings']==b['optimizer_settings'],'Executed seed pairing mismatch')
    write(RESULT/'run_ledger.json',{'complete':True,'planned_runs':15,'completed_runs':15,'runs':ledger})
    ledger_hash=digest(RESULT/'run_ledger.json')
    (RESULT/'run_ledger.json.sha256').write_text(ledger_hash+'  run_ledger.json\n')
    print('ALL_15_SEALED',ledger_hash,flush=True)
    # Cross-method analysis can be invoked only after this ledger is sealed.
    subprocess.run([sys.executable,str(HERE/'aggregate_results.py'),'--results',str(RESULT)],cwd=ROOT,check=True)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        traceback.print_exc()
        if RESULT.exists() and not (RESULT/'execution_failure.json').exists():
            write(RESULT/'execution_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
                  'training_stopped':True,'retry_or_seed_replacement':False,'scientific_aggregation_complete':False})
        sys.exit(1)
