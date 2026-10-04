"""Independently reconstruct sealed successful final metrics without TEST access."""
import argparse
import json
from pathlib import Path
from output_store import read, sha, require_completion, reconstruct
from verify_bundle import verify, manifest

def verify_results(output):
    verify();output=output.resolve()
    scope=manifest(output,'RESULT_SHA256SUMS')
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual!=set(scope)|{'RESULT_SHA256SUMS','RESULT_SHA256SUMS.sha256'}:raise ValueError('UNSEALED_RESULT_SCOPE')
    for rel,h in scope.items():
        if sha(output/rel)!=h:raise ValueError('RESULT_BYTE_MISMATCH: '+rel)
    if read(output/'result_manifest.json')['status']!='COMPLETE_FINAL_EVALUATION':raise ValueError('INCOMPLETE_OUTPUT_IS_NOT_FINAL')
    if read(output/'evaluation_failures.json')!={'status':'COMPLETE','failures':[]}:raise ValueError('FAILED_EVALUATION_IS_NOT_FINAL')
    rows=[json.loads(line) for line in (output/'eval_scenario_processing_ledger.jsonl').read_text().splitlines()];completed={}
    here=Path(__file__).resolve().parent
    for row in rows:
        sid=row['scenario_id'];p=output/row['metric_artifact']
        if sid in completed or sha(p)!=row['metric_sha256']:raise ValueError('SCENARIO_LEDGER_INTEGRITY_FAILED')
        record=read(p)
        if record['scenario_id']!=sid or row['numeric_artifact']!=record['numeric_artifact'] or row['numeric_sha256']!=record['numeric_sha256'] or row['eligible_ticks']!=record['eligible_ticks'] or row['metadata']!=record['metadata'] or row['partition']!='FINAL_EVAL' or row['tick_0_excluded'] is not True:raise ValueError('SCENARIO_LEDGER_BINDING_FAILED')
        completed[sid]=record
    require_completion(completed,set((here/'final_evaluation.txt').read_text().splitlines()),read(output/'archive_verification_receipt.json'),read(output/'access_ledger.json'))
    started=read(output/'attempt_started.json')
    if started['bundle_manifest_sha256']!=sha(here/'BUNDLE_SHA256SUMS') or started['threshold_sha256']!=sha(here/'conformal_thresholds.json'):raise ValueError('EXECUTED_BUNDLE_BINDING_FAILED')
    for name,obj in reconstruct(output,completed).items():
        if read(output/name)!=obj:raise ValueError('FINAL_METRIC_RECONSTRUCTION_FAILED: '+name)
    return {'status':'PASSED','scenarios':400,'scorers':15,'TEST_reaccess':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    print(json.dumps(verify_results(parser.parse_args().output),sort_keys=True))
