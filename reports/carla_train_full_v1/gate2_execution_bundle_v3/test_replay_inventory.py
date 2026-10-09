"""Real production-schema regression and fail-closed replay evidence binding."""
import ast
import copy
import shutil
import stat
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import gate2_common as c
from gate2_replay_inventory import ReplayInventory
from synthetic_replay_inventory_fixtures import CanonicalFixture

SCRATCH = None
SCHEMA_EVIDENCE = None


def scientific_code_proof():
    old_tree = ast.parse((c.V2/'gate2_pseudo.py').read_text())
    new_tree = ast.parse((c.BUNDLE/'gate2_pseudo.py').read_text())
    dump = lambda node:ast.dump(node,include_attributes=False)
    functions = ('generate_tick','write_jsonl_gzip','read_pseudo')
    unchanged = {}
    for name in functions:
        old = next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new = next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        unchanged[name] = dump(old)==dump(new)
    old_cls = next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='CALSink')
    new_cls = next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='CALSink')
    for name in ('member_begin','member_data','member_complete','close'):
        old = next(n for n in old_cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new = next(n for n in new_cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        unchanged['CALSink.'+name] = dump(old)==dump(new)
    old = copy.deepcopy(next(n for n in old_cls.body if isinstance(n,ast.FunctionDef) and n.name=='scenario_complete'))
    new = copy.deepcopy(next(n for n in new_cls.body if isinstance(n,ast.FunctionDef) and n.name=='scenario_complete'))
    c.require(isinstance(new.body[0],ast.Expr) and 'replay_inventory.bind' in ast.unparse(new.body[0]),
              'Missing v3 independent pre-consumption record check')
    new.body.pop(0)
    class ReplaceEngineeringBinding(ast.NodeTransformer):
        def visit_Subscript(self,node):
            if ast.unparse(node)=="self.current['record_sha256']":
                return ast.parse('self.scenario_inventory_record_sha256',mode='eval').body
            return self.generic_visit(node)
    old = ReplaceEngineeringBinding().visit(old)
    unchanged['CALSink.scenario_complete_scientific_body_after_exact_engineering_edits'] = dump(old)==dump(new)
    c.require(all(unchanged.values()),'Pseudo science or full replay/member/source checks changed')
    names = ('gate2_science.py','gate2_audit.py','execute_gate2.py','gate2_data.py','gate2_resolution.py',
             'synthetic_storage_fixtures.py','full_train_preregistered_protocol.json','development_gate_spec.json')
    byte_identical = {name:(c.BUNDLE/name).read_bytes()==(c.V2/name).read_bytes() for name in names}
    c.require(all(byte_identical.values()),'Frozen scientific/storage implementation changed')
    before = c.read_json(c.V2/'execution_bindings.json')
    after = c.read_json(c.BUNDLE/'execution_bindings.json')
    science_keys = ('HEAD','source_identity','membership','active_modalities','dimensions','window_ticks',
        'bootstrap_members','bootstrap_seed','pseudo_rotation','active_pseudo_severities','max_pseudo_per_tick',
        'pseudo_base_seed','GNSS_downstream','protocol_hashes','scientific_hashes','environment','environment_sha256',
        'prediction_semantics','preprocessing_policy_sha256','extra_source_hashes','expected_storage_counts')
    same_parameters = {key:before[key]==after[key] for key in science_keys}
    c.require(all(same_parameters.values()),'Scientific parameter/membership/environment changed')
    return {'success':True,'byte_identical_to_v2':byte_identical,'unchanged_pseudo_scientific_AST':unchanged,
            'scientific_parameters_and_environment_unchanged':same_parameters,
            'pseudo_source_v2_sha256':c.hash_file(c.V2/'gate2_pseudo.py'),
            'pseudo_source_v3_sha256':c.hash_file(c.BUNDLE/'gate2_pseudo.py'),
            'allowed_pseudo_amendment':'Canonical inventory admission/revalidation and replacement of the defective CAL input hash expression only'}


class CanonicalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = SCRATCH/'canonical-production-schema-fixture'
        root.mkdir()
        cls.fixture = CanonicalFixture(root)

    def setUp(self):
        self.index = copy.deepcopy(self.fixture.index)
        self.records = copy.deepcopy(self.fixture.records)
        self.legacy = self.fixture.legacy
        self.binding = self.fixture.binding

    def admitted(self, index=None, files=()):
        return ReplayInventory(self.index if index is None else index,self.binding,self.legacy,files)

    def resign_index(self):
        self.index['scenario_index_sha256'] = self.legacy.digest(self.index['scenarios'])

    def reject_index(self):
        with self.assertRaises(c.ReviewRequired):
            admitted = self.admitted()
            for ordinal,raw in enumerate(self.records,1):
                admitted.bind(raw,ordinal)
            admitted.finish(list(range(1,len(self.records)+1)))

    def reject_record(self):
        with self.assertRaises(c.ReviewRequired):
            self.admitted().bind(self.records[0],1)

    def bound_files(self):
        root = SCRATCH/uuid.uuid4().hex
        shutil.copytree(self.fixture.directory,root)
        seal = c.seal_tree(root)
        c.verify_seal(root,seal)
        files = tuple((root/name,c.hash_file(root/name)) for name in
                      ('source_inventory.json','inventory_scenarios.jsonl.gz','SHA256SUMS','SHA256SUMS.sha256'))
        return root,self.admitted(files=files)

    def test_mandatory_old_v2_exact_expression_fails_real_schema_v3_succeeds(self):
        global SCHEMA_EVIDENCE
        raw = self.records[0]
        entry = self.index['scenarios'][0]
        self.assertNotIn('record_sha256',raw)
        self.assertIn('record_sha256',entry)
        self.assertEqual(entry,dict(self.legacy.summary(raw),record_sha256=self.legacy.digest(raw)))
        path = c.V2/'gate2_pseudo.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CALSink')
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='scenario_complete')
        old = next(n for n in ast.walk(fn) if isinstance(n,ast.Subscript) and
                   isinstance(n.slice,ast.Constant) and n.slice.value=='record_sha256')
        self.assertEqual(ast.unparse(old),"self.current['record_sha256']")
        with self.assertRaisesRegex(KeyError,'record_sha256'):
            eval(compile(ast.Expression(body=old),str(path)+'::preserved-v2-line','eval'),
                 {'self':types.SimpleNamespace(current=raw)})
        exact = self.admitted().bind(raw,1)
        self.assertEqual(exact,self.legacy.digest(raw))
        self.assertEqual(exact,entry['record_sha256'])
        SCHEMA_EVIDENCE = {'success':True,'fixture_generated_by_exact_sealed_build_inventory':True,
            'raw_ledger_sample':raw,'index_schema':self.index['schema'],'index_summary_sample':entry,
            'raw_ledger_has_record_sha256':False,'index_summary_has_record_sha256':True,
            'old_v2_source_sha256':c.hash_file(path),'old_v2_expression':ast.unparse(old),
            'old_v2_observed_exception':"KeyError('record_sha256')",'v3_canonical_digest':exact,
            'indexed_record_sha256':entry['record_sha256'],'canonical_summary_plus_digest_exactly_matches':True,
            'sealed_inventory_implementation_sha256':c.hash_file(c.V8/'local_inventory.py'),
            'synthetic_only':True,'real_fitting_executed':False,'real_calibration_executed':False,
            'real_pseudo_generated':False,'graph_constructed':False,'GAT_executed':False,
            'TEST_requests':0,'network_requests':0}

    def test_actual_sealed_builder_and_verifier_roundtrip_all_raw_records(self):
        verified = self.legacy.verify_inventory(self.fixture.directory,self.binding)
        self.assertEqual(verified,self.index)
        admitted = self.admitted()
        for i,raw in enumerate(self.records,1):
            self.assertEqual(admitted.bind(raw,i),self.index['scenarios'][i-1]['record_sha256'])
        admitted.finish([1,2])

    def test_canonical_digest_is_existing_formatted_JSON_algorithm(self):
        raw = self.records[0]
        self.assertNotEqual(self.legacy.digest(raw),c.digest(c.canonical(raw)))
        self.assertEqual(self.admitted().bind(raw,1),self.legacy.digest(raw))

    def test_incorrect_indexed_record_sha256(self):
        self.index['scenarios'][0]['record_sha256'] = '0'*64
        self.resign_index()
        self.reject_index()

    def test_incorrect_index_archive_ordinal(self):
        self.index['scenarios'][0]['archive_ordinal'] = 2
        self.resign_index()
        self.reject_index()

    def test_incorrect_raw_archive_ordinal(self):
        self.records[0]['archive_ordinal'] = 2
        self.reject_record()

    def test_scenario_mismatch(self):
        self.index['scenarios'][0]['scenario_id'] = 'Town03/scenario-3'
        self.resign_index()
        self.reject_index()

    def test_first_TAR_offset_mismatch(self):
        self.index['scenarios'][0]['first_tar_offset'] += 512
        self.resign_index()
        self.reject_index()

    def test_parser_member_chain_mismatch(self):
        self.index['scenarios'][0]['parser_member_chain_sha256'] = '0'*64
        self.resign_index()
        self.reject_index()

    def test_source_member_set_mismatch(self):
        self.index['scenarios'][0]['source_member_set_sha256'] = '0'*64
        self.resign_index()
        self.reject_index()

    def test_modified_full_ledger_member_record(self):
        self.records[0]['members'][0]['sha256'] = '0'*64
        self.reject_record()

    def test_modified_full_ledger_excluded_metadata_or_readability_still_changes_digest(self):
        for field in ('metadata_integrity','image_readability'):
            with self.subTest(field=field):
                original = copy.deepcopy(self.records[0])
                self.records[0][field] = {'modified':True}
                self.reject_record()
                self.records[0] = original

    def test_modified_index_summary_ticks_rows_or_extra_field(self):
        for field,value in (('n_ticks',16),('emitted_rows',12),('town','Town03'),('unreviewed_extra',True)):
            with self.subTest(field=field):
                saved = copy.deepcopy(self.index)
                self.index['scenarios'][0][field] = value
                self.resign_index()
                self.reject_index()
                self.index = saved

    def test_duplicate_index_scenario(self):
        self.index['scenarios'][1]['scenario_id'] = self.index['scenarios'][0]['scenario_id']
        self.resign_index()
        self.reject_index()

    def test_duplicate_or_swapped_raw_scenario_alignment(self):
        admitted = self.admitted()
        admitted.bind(self.records[0],1)
        with self.assertRaises(c.ReviewRequired):
            admitted.bind(self.records[0],2)
        with self.assertRaises(c.ReviewRequired):
            admitted.bind(self.records[1],1)

    def test_missing_index_entry(self):
        self.index['scenarios'].pop()
        self.resign_index()
        self.reject_index()

    def test_shortened_self_consistent_index_rejects_remaining_raw_or_coverage(self):
        self.index['scenarios'].pop()
        self.index['scenario_count'] = 1
        self.resign_index()
        self.reject_index()

    def test_extra_index_entry(self):
        extra = copy.deepcopy(self.index['scenarios'][1])
        extra.update(archive_ordinal=3,scenario_id='Town03/scenario-3')
        self.index['scenarios'].append(extra)
        self.index['scenario_count'] = 3
        self.resign_index()
        self.reject_index()

    def test_extra_index_entry_with_original_count(self):
        self.index['scenarios'].append(copy.deepcopy(self.index['scenarios'][0]))
        self.resign_index()
        self.reject_index()

    def test_malformed_hashes_fail_closed(self):
        for value in ('0'*63,'G'*64,'A'*64,None,42,'0'*64+'\n'):
            with self.subTest(hash=value):
                saved = copy.deepcopy(self.index)
                self.index['scenarios'][0]['record_sha256'] = value
                self.resign_index()
                self.reject_index()
                self.index = saved

    def test_wrong_inventory_schema(self):
        self.index['schema'] = 'unreviewed-inventory'
        self.reject_index()

    def test_wrong_raw_record_schema(self):
        self.records[0]['schema'] = 'unreviewed-ledger'
        self.reject_record()

    def test_caller_provided_record_sha256_is_never_trusted_or_inserted(self):
        saved = copy.deepcopy(self.records[0])
        for value in ('0'*64,self.index['scenarios'][0]['record_sha256']):
            self.records[0]['record_sha256'] = value
            self.reject_record()
        self.assertNotIn('record_sha256',saved)
        self.assertNotIn('record_sha256',self.fixture.records[0])

    def test_index_admission_mutation_fails_closed(self):
        admitted = self.admitted()
        admitted.index['scenarios'][0]['record_sha256'] = '0'*64
        with self.assertRaisesRegex(c.ReviewRequired,'admission changed'):
            admitted.bind(self.records[0],1)

    def test_changed_inventory_ledger_file_hash(self):
        root,admitted = self.bound_files()
        (root/'inventory_scenarios.jsonl.gz').write_bytes(b'changed ledger')
        with self.assertRaisesRegex(c.ReviewRequired,'ledger/index/seal changed'):
            admitted.bind(self.records[0],1)

    def test_changed_inventory_index_file_hash(self):
        root,admitted = self.bound_files()
        path = root/'source_inventory.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(c.ReviewRequired,'ledger/index/seal changed'):
            admitted.bind(self.records[0],1)

    def test_changed_inventory_listing_or_detached_seal(self):
        for name in ('SHA256SUMS','SHA256SUMS.sha256'):
            with self.subTest(seal=name):
                root,admitted = self.bound_files()
                (root/name).write_bytes(b'changed seal')
                with self.assertRaisesRegex(c.ReviewRequired,'ledger/index/seal changed'):
                    admitted.bind(self.records[0],1)

    def test_ledger_path_escape_or_alias_fails_before_read(self):
        for name in ('../escape.gz','C:/escape.gz','inventory_scenarios.jsonl.gz/../alias.gz'):
            self.index['ledger_name'] = name
            with patch.object(Path,'open',side_effect=AssertionError('untrusted path must not be read')):
                with self.assertRaises(c.ReviewRequired):
                    self.admitted()

    def test_linked_inventory_path_rejected(self):
        root,admitted = self.bound_files()
        target = root/'inventory_scenarios.jsonl.gz'
        original = Path.resolve
        def linked(path,*args,**kwargs):
            return root/'escaped' if path==target else original(path,*args,**kwargs)
        with patch.object(Path,'resolve',linked):
            with self.assertRaisesRegex(c.ReviewRequired,'Linked/junction path rejected'):
                admitted.bind(self.records[0],1)

    def test_reparse_junction_inventory_path_rejected(self):
        root,admitted = self.bound_files()
        target = root/'inventory_scenarios.jsonl.gz'
        original = Path.lstat
        def reparse(path,*args,**kwargs):
            value = original(path,*args,**kwargs)
            return types.SimpleNamespace(st_mode=value.st_mode,
                st_file_attributes=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024)) if path==target else value
        with patch.object(Path,'lstat',reparse):
            with self.assertRaisesRegex(c.ReviewRequired,'Reparse path rejected'):
                admitted.bind(self.records[0],1)

    def test_full_record_mutation_after_scenario_begin_rejected_before_generation(self):
        from gate2_pseudo import CALSink
        fake = types.SimpleNamespace(evidence={'membership':{'FIT_NORMAL':[self.records[0]['scenario_id']],
                                                             'CAL_NORMAL':[self.records[1]['scenario_id']]}},
                                     verify=lambda:{'committed':{}})
        sink = CALSink(fake,{},self.records,self.binding,self.admitted())
        raw = self.records[0]
        sink.scenario_begin({'scenario_id':raw['scenario_id'],'archive_ordinal':1,
                             'first_tar_header_offset':raw['first_tar_offset']})
        self.assertNotIn('record_sha256',sink.current)
        sink.current['metadata_integrity'] = {'modified':True}
        with patch('gate2_pseudo.generate_tick',side_effect=AssertionError('must fail before pseudo')) as generate:
            with self.assertRaises(c.ReviewRequired):
                sink.scenario_complete({'scenario_id':raw['scenario_id']})
            self.assertEqual(generate.call_count,0)

    def test_incomplete_or_extra_replay_coverage_fails_closed(self):
        for ordinals in ([1],[1,2,3],[2,1],[1,1]):
            with self.assertRaises(c.ReviewRequired):
                self.admitted().finish(ordinals)


class HistoricalV3Tests(unittest.TestCase):
    def test_scientific_code_parameters_and_pseudo_body_unchanged_except_exact_binding(self):
        self.assertTrue(scientific_code_proof()['success'])

    def test_v2_stopped_execution_seals_and_raw_workspace_preserved(self):
        from gate2_forensics import verify_v2_failure,V2_FIT_SEALS
        result = verify_v2_failure()
        self.assertEqual(result,c.read_json(c.BUNDLE/'v2_failure_evidence.json'))
        self.assertEqual(result['committed_units'],V2_FIT_SEALS)
        self.assertEqual(result['raw_files'],1288)
        self.assertFalse(result['scientific_validity_inferred'])

    def test_new_v3_runtime_manifest_cannot_rebind_v2_FIT_units(self):
        from gate2_runtime import manifest_for
        bindings = c.read_json(c.BUNDLE/'execution_bindings.json')
        self.assertEqual(bindings['real_runtime'],'reports/carla_train_full_v1/gate2_train_health_v3')
        self.assertEqual(bindings['FIT_policy'],'Recompute frozen FIT under v3 manifest; no cross-runtime adoption')
        old = c.read_json(c.V2_RUNTIME/'run_manifest.json')
        evidence = {'bindings':bindings,'environment':old['environment'],'membership':old['membership'],
            'source_binding':old['source_binding'],'completion':old['upstream_completion'],
            'block_resolutions':old['block_resolutions']}
        current = manifest_for('f'*64,evidence)
        self.assertNotEqual(c.digest(c.canonical(current)),c.hash_file(c.V2_RUNTIME/'run_manifest.json'))
        for name in ('fit-Camera','fit-Seg','fit-IMU'):
            unit = c.read_json(c.V2_RUNTIME/'units'/name/'unit.json')
            self.assertNotEqual(unit['run_manifest_sha256'],c.digest(c.canonical(current)))
