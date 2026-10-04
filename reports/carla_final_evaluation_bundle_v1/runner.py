"""Preparation preflight and separately authorized FINAL_EVALUATION_ATTEMPT 001."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OUTPUT=Path('/kaggle/working/cognix_final_evaluation_attempt001_v1')
FINAL_EVALUATION_ATTEMPT='001'
CALIBRATION_SOURCE='successful Attempt003 v4'

def preflight(check_runtime=True):
    from verify_bundle import verify
    result=verify()
    if check_runtime:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        from scientific_runtime import runtime_check
        result['runtime']=runtime_check()
    return result

def execute():
    preflight()
    if OUTPUT.exists():raise RuntimeError('EXISTING_ATTEMPT_OR_OUTPUT_NO_RETRY_NO_RESUME')
    if HERE==OUTPUT or HERE.is_relative_to(OUTPUT) or OUTPUT.is_relative_to(HERE):raise RuntimeError('OUTPUT_MUST_BE_ISOLATED')
    from frozen_runtime import load
    from scientific_runtime import runtime_check
    from scientific_processor import EvalProcessor
    from output_store import save, require_completion, reconstruct, seal_results
    runtime=runtime_check();models,agents=load('cuda:0')
    evaluation=frozenset((HERE/'final_evaluation.txt').read_text().splitlines())
    cal=frozenset((HERE/'final_conformal_cal.txt').read_text().splitlines())
    prior=frozenset((HERE/'prior_exposed_cal.txt').read_text().splitlines())
    exclusions=frozenset((HERE/'historical_exclusions.txt').read_text().splitlines())
    OUTPUT.mkdir(parents=True,exist_ok=False)
    save(OUTPUT/'attempt_started.json',{'FINAL_EVALUATION_ATTEMPT':FINAL_EVALUATION_ATTEMPT,'CALIBRATION_SOURCE':CALIBRATION_SOURCE,
      'single_attempt':True,'automatic_retry':False,'started_unix':time.time(),'runtime':runtime,
      'bundle_manifest_sha256':hashlib.sha256((HERE/'BUNDLE_SHA256SUMS').read_bytes()).hexdigest(),
      'threshold_sha256':hashlib.sha256((HERE/'conformal_thresholds.json').read_bytes()).hexdigest()})
    processor=EvalProcessor(models,agents,evaluation,OUTPUT,'cuda:0')
    ledger={'http_requests':0,'range_requests':0,'automatic_retry':False,'raw_archive_retained':False,
      'fresh_cal_scenarios_decoded':0,'prior_exposed_cal_scenarios_decoded':0,'historical_excluded_scenarios_decoded':0,
      'final_eval_only_science':True,'FINAL_EVALUATION_ATTEMPT':FINAL_EVALUATION_ATTEMPT,'CALIBRATION_SOURCE':CALIBRATION_SOURCE}
    try:
        from transport import open_one_shot
        from streaming import scan
        with open_one_shot(ledger) as response:
            receipt=scan(response,evaluation,cal,exclusions,processor,ledger,91538225599,prior_exposed_cal=prior)
        # No official aggregate metric artifact is produced before these gates.
        require_completion(processor.completed,evaluation,receipt,ledger)
        save(OUTPUT/'archive_verification_receipt.json',receipt)
        final=reconstruct(OUTPUT,processor.completed)
        for name,obj in final.items():save(OUTPUT/name,obj)
        save(OUTPUT/'zero_protected_role_decode_receipt.json',{'fresh_cal_scenarios_decoded':0,
          'prior_exposed_cal_scenarios_decoded':0,'historical_excluded_scenarios_decoded':0,
          'decoder_entry_policy':'FINAL_EVAL gate in scanner, processor and metric sink','limitation':'Executed-code ledger; not an OS-wide access proof'})
        save(OUTPUT/'evaluation_failures.json',{'status':'COMPLETE','failures':[]})
        save(OUTPUT/'result_manifest.json',{'status':'COMPLETE_FINAL_EVALUATION','eval_scenarios':400,'scorers':15,
          'calibration_source':CALIBRATION_SOURCE,'FINAL_EVALUATION_ATTEMPT':FINAL_EVALUATION_ATTEMPT,
          'threshold_refit':False,'no_best_seed_selection':True,'prediction_ensemble':False,'raw_TEST_frames_retained':False,
          'metric_specs':json.loads((HERE/'protocol_bindings.json').read_text())['frozen_spec_bindings'],
          'undefined_results_retained':True,'numeric_intermediates_retained':True})
    except BaseException as exc:
        save(OUTPUT/'evaluation_failures.json',{'status':'INCOMPLETE_NO_RETRY','exception':type(exc).__name__,
          'message':str(exc),'traceback':traceback.format_exc(),'eval_completed':len(processor.completed)})
        save(OUTPUT/'result_manifest.json',{'status':'INCOMPLETE_NO_RETRY','partial_results_are_not_final':True,
          'automatic_retry':False,'must_not_retry_or_resume':True})
        raise
    finally:
        ledger['eval_scenarios_completed']=len(processor.completed)
        save(OUTPUT/'access_ledger.json',ledger)
        seal_results(OUTPUT)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--execute-separately-authorized-final-evaluation',action='store_true')
    args=parser.parse_args()
    if args.preflight and args.execute_separately_authorized_final_evaluation:parser.error('Choose one action')
    if args.execute_separately_authorized_final_evaluation:execute()
    else:print(json.dumps(preflight(),sort_keys=True))
