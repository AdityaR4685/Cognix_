"""Read-only v3 preflight: the exact shared pre-source production closure."""
import argparse
from graph_common import *
from execution_closure import pre_source_closure,complete_source_integrity


def verify_preflight(external_seal,*,full_source_hash=True,progress=lambda message:None):
    closure=pre_source_closure(external_seal,progress=progress)
    source=complete_source_integrity(closure,progress) if full_source_hash else None
    tests=read_json(BUNDLE/'synthetic_test_results.json')
    actual={p.name:hash_file(p) for p in BUNDLE.glob('*.py')}
    require(tests['failed']==tests['errors']==tests['skipped']==0 and tests['passed']==tests['tests_run'] and
        tests['bundle_python_sha256']==actual and tests['real_activity']==ZERO,'Accepted tested source/evidence mismatch')
    require(all(p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY for p in BUNDLE.rglob('*') if p.is_file()),
        'V3 bundle is not immutable/read-only')
    require(read_json(BUNDLE/'preparation_audit.json')['scientific_invariance']=='PASS','Preparation invariance failed')
    future_namespaces_absent()
    return dict(status='READ_ONLY_PREFLIGHT_PASS',bundle_path=str(BUNDLE),bundle_seal=external_seal,
        shared_pre_source_closure='execution_closure.pre_source_closure',source_integrity=source,
        real_context_status=closure['real_context_status'],
        clean_counts={k:closure['production_clean_resolution'][k] for k in
            ('total_loaded','native_count','adopted_count','FIT_loaded','CAL_loaded','GRAPH_TRAIN_loaded','GRAPH_VAL_loaded')},
        inventory_records_checked=closure['production_inventory_schema']['production_records_checked'],
        canonical_digests_checked=closure['production_inventory_schema']['canonical_digests_checked'],
        fake_record_sha256_dependency=False,disk_resources=closure['disk_resources'],
        execution_order='PASS',historical_preservation=closure['historical_preservation'],git_state=closure['git_state'],
        data_v3_exists=False,pending_v3_exists=False,runs_v3_exists=False,files_written=0,
        tests={k:tests[k] for k in ('passed','failed','errors','skipped')},activity=dict(ZERO))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle-seal',required=True);args=parser.parse_args()
    counters=install_guard('preflight',source_integrity=True)
    result=verify_preflight(args.bundle_seal,progress=lambda message:print(message,flush=True))
    result['guard_counters']={k:v for k,v in counters.items() if k!='admitted_scientific_git_reads'}
    result['authenticated_scientific_git_reads']=len(counters['admitted_scientific_git_reads'])
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
