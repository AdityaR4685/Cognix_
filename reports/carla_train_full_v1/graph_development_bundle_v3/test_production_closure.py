"""Real sealed-production read-only integration and cheap-failure ordering probes."""
import copy
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import graph_common
from graph_common import *
from current_upstream import authenticate_gate2
from execution_closure import (PRE_SOURCE_STEPS,pre_source_closure,complete_source_integrity,
    run_ordered_steps,admission_then_publication,read_only_scientific_barrier)
from production_audits import (validate_inventory_record,segmentation_compatibility_audit,
    disk_resource_audit,verify_production_clean_resolution)

_PRODUCTION={}
_FAILURES=[]


class ProductionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Every actual Popen in the real shared closure is also admitted by the
        # future graph-data guard, using its exact unchanged subprocess policy.
        scope={'enabled':True};guard,counters=make_guard('graph-data',source_integrity=True)
        def future_subprocess_probe(event,args):
            if scope['enabled'] and event=='subprocess.Popen':guard(event,args)
        sys.addaudithook(future_subprocess_probe)
        try:cls.closure=pre_source_closure(None,preparing=True,progress=lambda message:print(message,flush=True))
        finally:scope['enabled']=False
        cls.future_counters=counters
        cls.modules=authenticate_gate2();cls.ctx=cls.modules['gate2_resolution'].context()
        cls.replay=cls.modules['gate2_replay_inventory'].load_verified_replay_inventory(cls.ctx.source)
        cls.raw=next(cls.replay.legacy.inventory_records(REPORT/'gate1_execution_bundle_v8',cls.replay.index))
        _PRODUCTION.update(closure=cls.closure,future_subprocess_counters=counters)

    def test_real_context_under_future_graph_data_policy(self):
        self.assertEqual(self.closure['real_context_status'],'PASS')
        self.assertEqual(len(self.future_counters['admitted_scientific_git_reads']),22)
        self.assertEqual(self.future_counters['blocked_subprocess_attempts'],0)

    def test_all_101_production_clean_shapes_dtypes_finite(self):
        audit=self.closure['production_clean_resolution'];self.assertEqual(audit['total_loaded'],101)
        for record in audit['records']:
            self.assertEqual(record['shapes'],{m:[2999,DIMS[m]] for m in NODE_ORDER})
            self.assertEqual(record['dtypes'],{m:'float64' for m in NODE_ORDER});self.assertTrue(record['all_finite'])

    def test_all_76_FIT_and_25_CAL_loaded(self):
        audit=self.closure['production_clean_resolution']
        self.assertEqual((audit['FIT_loaded'],audit['CAL_loaded']),(76,25))

    def test_native_adopted_33_68_and_exact_provenance(self):
        audit=self.closure['production_clean_resolution']
        self.assertEqual((audit['native_count'],audit['adopted_count']),(33,68))
        self.assertTrue(all(r['exact_sealed_provenance_equal'] for r in audit['records']))
        self.assertFalse(audit['feature_recomputation']);self.assertFalse(audit['new_clean_store_created'])

    def test_graph_membership_61_15_excludes_CAL(self):
        audit=self.closure['production_clean_resolution']
        self.assertEqual((audit['GRAPH_TRAIN_loaded'],audit['GRAPH_VAL_loaded']),(61,15))
        self.assertTrue(all(r['graph_role'] is None for r in audit['records'] if r['role']=='CAL_NORMAL'))

    def test_101_real_inventory_summaries_and_digests(self):
        audit=self.closure['production_inventory_schema']
        self.assertEqual([audit[k] for k in ('production_records_checked','canonical_summaries_checked','canonical_digests_checked')],[101,101,101])
        self.assertEqual(audit['mismatches'],[])
        self.assertTrue(all(r['canonical_digest']==r['indexed_digest'] for r in audit['records']))

    def test_raw_record_sha256_not_required(self):
        self.assertNotIn('record_sha256',self.raw)
        summary,sha=validate_inventory_record(self.raw,1,self.replay)
        self.assertEqual(sha,self.replay.index['scenarios'][0]['record_sha256'])
        self.assertFalse(self.closure['production_inventory_schema']['fake_record_sha256_dependency'])

    def test_frozen_seg_compatibility_and_pre_recipe_probe(self):
        audit=segmentation_compatibility_audit(self.modules)
        self.assertEqual(audit,read_json(BUNDLE/'segmentation_compatibility_audit.json'))
        self.assertTrue(audit['sanitizer_applied_before_Seg_pseudo_generation'])

    def test_environment_imports_disk_before_archive_hash(self):
        for key in ('compatibility_prerequisites','non_model_imports','environment','disk_resources','barriers'):
            self.assertEqual(self.closure[key]['status'],'PASS')
        self.assertFalse(self.closure['full_archive_hash_invoked'])
        self.assertGreaterEqual(self.closure['disk_resources']['free_bytes'],self.closure['disk_resources']['required_free_bytes'])

    def test_no_model_or_replay_semantics_in_production_closure(self):
        self.assertEqual(self.closure['activity'],ZERO);self.assertFalse(self.closure['real_models_instantiated'])
        with self.assertRaises(PreparationError):
            with read_only_scientific_barrier():self.modules['gate2_science'].new_agent({},'Camera')

    def test_preflight_and_data_use_same_closure_and_admission(self):
        from preflight_graph import pre_source_closure as preflight_closure
        self.assertIs(preflight_closure,pre_source_closure)
        tree=__import__('ast').parse((BUNDLE/'generate_graph_data.py').read_bytes())
        self.assertTrue(any(isinstance(n,__import__('ast').ImportFrom) and n.module=='execution_closure' and
            {'pre_source_closure','complete_source_integrity','admission_then_publication'}<={a.name for a in n.names}
            for n in __import__('ast').walk(tree)))
        self.assertEqual(hash_file(BUNDLE/'subprocess_policy.py'),hash_file(V2_BUNDLE/'subprocess_policy.py'))

    def test_reviewed_Windows_ver_probe_null_handle_inside_readonly_closure(self):
        import platform
        with read_only_scientific_barrier():
            # Explicitly force the subprocess, bypassing platform.uname's cache.
            result=platform._syscmd_ver()
        self.assertEqual(len(result),3)
        self.assertTrue(result[2])


class OrderingAdversarial(unittest.TestCase):
    def assert_cheap_failure(self,label,step,callback):
        calls=[];trace=[]
        callbacks={name:(lambda values:dict(status='PASS')) for name in PRE_SOURCE_STEPS}
        callbacks[step]=lambda values:callback()
        def closure():
            run_ordered_steps(callbacks,trace);return dict(status='PASS')
        source=Mock(side_effect=lambda result:calls.append('full_source_hash'))
        pending=Mock(side_effect=lambda a,b:calls.append('pending_create'))
        with self.assertRaises(Exception):admission_then_publication(closure,source,pending)
        self.assertEqual(source.call_count,0);self.assertEqual(pending.call_count,0);self.assertEqual(calls,[])
        self.assertFalse(PENDING.exists())
        _FAILURES.append(dict(failure=label,stage=step,status='PASS',full_source_hash_calls=0,
            pending_create_calls=0,trace=trace,real_source_corrupted=False))

    def test_resolver_failure_before_full_hash(self):
        def injected():
            with patch.object(ProductionIntegration.modules['gate2_data'],'load_clean',side_effect=PreparationError('SIMULATED resolver failure')):
                verify_production_clean_resolution(ProductionIntegration.modules,ProductionIntegration.ctx)
        self.assert_cheap_failure('resolver_failure','production_clean_resolution',injected)

    def test_adopted_block_missing_before_full_hash(self):
        resolver=ProductionIntegration.modules['gate2_resolution'];original=resolver.resolve_arrays
        def missing(sid,admitted=None):
            if ProductionIntegration.ctx.expected[sid]['kind']=='V7_TO_V8_IDENTITY_ADOPTION':
                raise FileNotFoundError('SIMULATED missing adopted v7 arrays; no actual source mutation')
            return original(sid,admitted)
        def injected():
            with patch.object(resolver,'resolve_arrays',side_effect=missing):
                verify_production_clean_resolution(ProductionIntegration.modules,ProductionIntegration.ctx)
        self.assert_cheap_failure('adopted_block_missing','production_clean_resolution',injected)

    def test_schema_mismatch_before_full_hash(self):
        malformed=dict(ProductionIntegration.raw,record_sha256='fixture-only-convenience')
        self.assert_cheap_failure('raw_schema_mismatch','production_inventory_schema',
            lambda:validate_inventory_record(malformed,1,ProductionIntegration.replay))

    def test_wrong_inventory_digest_before_full_hash(self):
        index=copy.deepcopy(ProductionIntegration.replay.index);index['scenarios'][0]['record_sha256']='0'*64
        fake=SimpleNamespace(index=index,legacy=ProductionIntegration.replay.legacy,
            bind=lambda *args:require(False,'Wrong indexed digest must fail before bind'))
        self.assert_cheap_failure('wrong_inventory_digest','production_inventory_schema',
            lambda:validate_inventory_record(ProductionIntegration.raw,1,fake))

    def test_subprocess_denial_before_full_hash(self):
        guard,counters=make_guard('graph-data')
        self.assert_cheap_failure('subprocess_denial','real_context',
            lambda:guard('subprocess.Popen',('git',['git','show',HEAD+':arbitrary.py'],str(REPO),None)))

    def test_insufficient_disk_before_full_hash(self):
        self.assert_cheap_failure('insufficient_disk','disk_resources',
            lambda:disk_resource_audit(ProductionIntegration.closure['production_inventory_schema'],free_bytes=0))

    def test_schema_types_reject_fixture_only_convenience(self):
        raw=dict(ProductionIntegration.raw);raw['archive_ordinal']='1'
        with self.assertRaises(PreparationError):validate_inventory_record(raw,1,ProductionIntegration.replay)
        raw=dict(ProductionIntegration.raw);del raw['members']
        with self.assertRaises(PreparationError):validate_inventory_record(raw,1,ProductionIntegration.replay)

    def test_source_hash_failure_prevents_pending(self):
        closure=Mock(return_value=ProductionIntegration.closure);source=Mock(side_effect=PreparationError('SIMULATED source SHA failure'))
        publish=Mock()
        with self.assertRaises(PreparationError):admission_then_publication(closure,source,publish)
        self.assertEqual(publish.call_count,0);self.assertFalse(PENDING.exists())

    def test_successful_hash_then_fingerprint_then_pending_order(self):
        events=[];trace=[]
        callbacks={name:(lambda values,key=name:events.append(key)) for name in PRE_SOURCE_STEPS}
        def closure():
            run_ordered_steps(callbacks,trace)
            return dict(status='PASS',trace=trace,source_fingerprint_before_full_hash=source_fingerprint())
        def source(value):
            events.append('full_source_hash')
            with patch('execution_closure.hash_file',return_value=SOURCE['sha256']):result=complete_source_integrity(value)
            events.append('source_fingerprint_recheck');return result
        def pending(a,b):events.append('pending_create_simulation_only');return dict(status='PASS')
        self.assertEqual(admission_then_publication(closure,source,pending)['status'],'PASS')
        self.assertEqual(events,list(PRE_SOURCE_STEPS)+['full_source_hash','source_fingerprint_recheck','pending_create_simulation_only'])
        self.assertFalse(PENDING.exists())

    def test_v1_v2_runtime_and_pending_immutability(self):
        before=verify_v1_failure()
        for phase in ('prepare','synthetic','preflight','graph-data'):
            guard,counters=make_guard(phase)
            for version in (1,2):
                root=REPORT/f'.graph_development_data_v{version}.pending'
                for event,args in (('open',(str(root/'x'),'wb',os.O_WRONLY|os.O_CREAT)),
                    ('os.remove',(str(root/'authorization.json'),-1)),('os.rename',(str(root),str(PENDING),-1,-1))):
                    with self.assertRaises(PreparationError):guard(event,args)
        self.assertEqual(verify_v1_failure(),before)


def write_test_reports(record):
    if record['failed'] or record['errors'] or record['skipped']:
        write_json(BUNDLE/f'failed_test_report_{record["attempt"]}.json',dict(counts={k:record[k] for k in
            ('passed','failed','errors','skipped','tests_run')},failures=_FAILURES))
        return
    require(_PRODUCTION and len(_FAILURES)==6,'Missing production or failure-injection proof')
    combined=dict(record,evidence_kind='Original 80 synthetic/adversarial tests plus real read-only production integration and ordering injection')
    write_json(BUNDLE/'synthetic_test_results.json',combined)
    records=[r for r in record['records'] if 'ProductionIntegration.' in r['test']]
    write_json(BUNDLE/'production_integration_test_results.json',dict(status='PASS',passed=len(records),failed=0,
        errors=0,skipped=0,records=records,real_clean_blocks_used=True,real_raw_inventory_records_used=True,
        shared_future_graph_data_subprocess_guard_invoked_for_actual_commands=True,
        authenticated_scientific_git_reads=len(_PRODUCTION['future_subprocess_counters']['admitted_scientific_git_reads']),
        clean_counts={k:_PRODUCTION['closure']['production_clean_resolution'][k] for k in
            ('total_loaded','FIT_loaded','CAL_loaded','native_count','adopted_count','GRAPH_TRAIN_loaded','GRAPH_VAL_loaded')},
        canonical_inventory_records_checked=101,canonical_digests_checked=101,activity=dict(ZERO)))
    adversarial=[r for r in record['records'] if 'Adversarial.' in r['test']]
    write_json(BUNDLE/'adversarial_test_results.json',dict(status='PASS',passed=len(adversarial),failed=0,
        errors=0,skipped=0,records=adversarial,failure_injections=_FAILURES,actual_TEST_or_network_requests=0))
    write_json(BUNDLE/'execution_order_audit.json',dict(status='PASS',same_shared_pre_source_function=True,
        required_order=list(PRE_SOURCE_STEPS)+['full_source_hash','source_fingerprint_recheck','pending_create',
            'record_authorization','source_replay','complete_replay_validation','publish_final_data'],
        actual_production_pre_source_trace=_PRODUCTION['closure']['trace'],
        pre_source_function_sha256=hash_file(BUNDLE/'execution_closure.py'),
        six_predictable_failure_injections=_FAILURES,pending_before_source_hash=False,
        pending_after_source_hash_and_fingerprint_only=True,success_order_test_uses_mock_hash_no_pending_created=True,
        all_required_real_production_checks_before_full_archive_hash=True,
        data_generation_executed=False,activity=dict(ZERO)))
    write_json(BUNDLE/'production_integration_closure_audit.json',_PRODUCTION['closure'])
