"""Both storage forms, byte identity, role mixing and adversarial readback."""
import ast
import copy
import os
import stat
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import gate2_common as c
import gate2_data as data
import gate2_resolution as resolution
from synthetic_storage_fixtures import StorageFixture

SCRATCH = None


class StorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = SCRATCH/'pristine-artificial-storage'
        root.mkdir()
        cls.pristine = StorageFixture(root)

    def setUp(self):
        root = SCRATCH / uuid.uuid4().hex
        root.mkdir()
        self.f = self.pristine.clone(root)
        self.sid = self.f.ids[0]
        self.adopted = self.f.store / 'verified_clean_blocks' / self.f.native.block_name(self.sid)
        self.old = self.f.paths[self.sid]
        self.native_sid = self.f.ids[2]
        self.native = self.f.paths[self.native_sid]

    def load(self, sid=None):
        return data.load_clean(self.sid if sid is None else sid, admitted=self.f.ctx)

    def reject(self, sid=None):
        with self.assertRaises(Exception):
            self.load(sid)

    def change_json(self, path, mutate, resign=False):
        record = c.read_json(path)
        mutate(record)
        if resign:
            record['payload_sha256'] = self.f.native.digest(record['payload'])
        path.write_bytes(c.canonical(record))

    def test_adoption_feature_bytes_identical_to_sealed_v7_and_no_feature_copy(self):
        before = {p.name:p.read_bytes() for p in self.old.iterdir()}
        record = self.load()
        direct = data.load_npz(self.old / 'arrays.npz')
        for key in direct:
            self.assertEqual(record['arrays'][key].dtype, direct[key].dtype)
            self.assertEqual(record['arrays'][key].tobytes(), direct[key].tobytes())
        self.assertEqual(record['resolved_provenance']['storage_form'], resolution.ADOPTED)
        self.assertEqual({p.name for p in self.adopted.iterdir()}, {'adoption.json','oov_ledger.jsonl.gz'})
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.old.iterdir()})

    def test_native_v8_matches_exact_preserved_v1_loader_feature_bytes(self):
        tree = ast.parse((c.V1 / 'gate2_data.py').read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'load_clean')
        ns = dict(STORE=self.f.store, MODALITIES=c.MODALITIES, np=np, require=c.require,
            read_json=c.read_json, hash_file=c.hash_file, array_digest=data.array_digest,
            load_npz=data.load_npz, check_features=data.check_features)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'sealed-v1-load-clean','exec'),ns)
        old = ns['load_clean'](self.native_sid)
        new = self.load(self.native_sid)
        for key in old['arrays']:
            self.assertEqual(old['arrays'][key].tobytes(),new['arrays'][key].tobytes())
        for key in ('scenario_id','source_split','role','synthetic_corruption',
                    'block_npz_sha256','block_content_sha256'):
            self.assertEqual(old[key],new[key])
        self.assertEqual(new['resolved_provenance']['storage_form'],resolution.NATIVE)

    def test_both_sealed_verifiers_independently_called_before_adopted_consumption(self):
        with (patch.object(self.f.adoption,'verify_adoption',wraps=self.f.adoption.verify_adoption) as adopt,
              patch.object(self.f.old,'verify_block',wraps=self.f.old.verify_block) as old):
            self.load()
        self.assertEqual(adopt.call_count,1)
        self.assertEqual(old.call_count,2)

    def test_no_extraction_decoding_or_feature_recomputation_for_either_form(self):
        import PIL.Image
        with (patch.object(self.f.native,'extract_clean',side_effect=AssertionError('no extraction')),
              patch.object(self.f.old,'extract_clean',side_effect=AssertionError('no extraction')),
              patch.object(PIL.Image,'open',side_effect=AssertionError('no decode'))):
            for sid in self.f.ids:
                record = self.load(sid)
                self.assertFalse(record['resolved_provenance']['feature_recomputed'])

    def test_mixed_FIT_CAL_both_storage_forms_exact_role_isolation(self):
        reads = []
        def reader(sid):
            reads.append(sid)
            return self.load(sid)
        for modality in c.MODALITIES:
            reads.clear()
            x,binding = data.assemble_fit(self.f.membership,modality,reader)
            self.assertEqual(reads,sorted(self.f.membership['FIT_NORMAL']))
            self.assertEqual(set(r['resolved_provenance']['storage_form'] for r in binding['clean_block_bindings']),
                             {resolution.NATIVE,resolution.ADOPTED})
            direct = np.concatenate([self.f.features[sid][modality.lower()] for sid in reads])
            self.assertEqual(x.tobytes(),direct.tobytes())
            reads.clear()
            def synthetic_pseudo(sid):
                rid = data.recipe_for_tick(1 if modality == 'Camera' else 2 if modality == 'Seg' else 5)
                tick = 1 if modality == 'Camera' else 2 if modality == 'Seg' else 5
                row = {'source_split':'train','partition':'CAL_NORMAL','parent_scenario':sid,
                    'parent_tick':tick,'window_start_tick':0,'window_end_tick':tick,'target_normal':0,
                    'recipe_id':rid,'severity':c.SEVERITIES[rid],'seed':tick,'modality':modality.lower(),
                    'synthetic_corruption':True,'graph_role':None,'upstream_fit':False}
                arrays = {m.lower():np.full((1,d),-10.,dtype=np.float64) for m,d in c.DIMS.items()}
                return arrays,[row]
            x,y,rows = data.modality_calibration(self.f.membership,modality,synthetic_pseudo,reader)
            self.assertEqual(reads,sorted(self.f.membership['CAL_NORMAL']))
            self.assertFalse(set(reads)&set(self.f.membership['FIT_NORMAL']))
            self.assertEqual(int((y==1).sum()),5998)
            self.assertEqual(int((y==0).sum()),2)
            self.assertEqual(len(rows),6000)

    def test_complete_mixed_resolution_manifest_counts_and_role_counts(self):
        report = resolution.verify_all_resolutions(self.f.ctx)
        self.assertEqual(report['counts'],{resolution.NATIVE:2,resolution.ADOPTED:2})
        for role in ('FIT_NORMAL','CAL_NORMAL'):
            self.assertEqual(report['role_counts'][role],{resolution.NATIVE:1,resolution.ADOPTED:1})
        self.assertFalse(report['feature_recomputed'])
        self.assertFalse(report['adopted_feature_files_copied'])

    def test_malformed_adoption_fails_closed(self):
        (self.adopted/'adoption.json').write_bytes(b'{malformed')
        self.reject()

    def test_duplicate_adoption_JSON_keys_fail_closed(self):
        raw = (self.adopted/'adoption.json').read_bytes()
        (self.adopted/'adoption.json').write_bytes(b'{"schema":"alias",'+raw[1:])
        self.reject()

    def test_changed_adoption_payload_hash_fails_closed(self):
        self.change_json(self.adopted/'adoption.json',lambda r:r.update(payload_sha256='0'*64))
        self.reject()

    def test_changed_adoption_schema_fails_closed(self):
        self.change_json(self.adopted/'adoption.json',lambda r:r.update(schema='unreviewed'))
        self.reject()

    def test_arbitrary_path_escape_never_followed_even_with_resigned_payload(self):
        escape = self.f.base/'escape'
        escape.mkdir()
        self.change_json(self.adopted/'adoption.json',lambda r:r['payload'].update(v7_block_path=str(escape)),True)
        opened=[]
        original=Path.open
        def tracked(path,*args,**kwargs):
            opened.append(path)
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',tracked):
            self.reject()
        self.assertFalse(any(p==escape or escape in p.parents for p in opened))

    def test_path_alias_dotdot_is_not_accepted(self):
        alias=str(self.old/'..'/self.old.name)
        self.change_json(self.adopted/'adoption.json',lambda r:r['payload'].update(v7_block_path=alias),True)
        self.reject()

    def test_unexpected_adoption_feature_files_fail_closed(self):
        (self.adopted/'arrays.json').write_bytes(b'{}')
        self.reject()

    def test_missing_adoption_ledger_fails_closed(self):
        (self.adopted/'oov_ledger.jsonl.gz').unlink()
        self.reject()

    def test_tampered_adoption_OOV_ledger_fails_closed(self):
        (self.adopted/'oov_ledger.jsonl.gz').write_bytes(b'corrupt')
        self.reject()

    def test_changed_v7_npz_fails_closed(self):
        (self.old/'arrays.npz').write_bytes(b'corrupt')
        self.reject()

    def test_changed_v7_scientific_manifest_fails_closed(self):
        self.change_json(self.old/'arrays.json',lambda r:r.update(window_ticks=13))
        self.reject()

    def test_changed_v7_immutable_manifest_fails_closed(self):
        self.change_json(self.old/'manifest.json',lambda r:r.update(payload_sha256='0'*64))
        self.reject()

    def test_unexpected_v7_inventory_fails_closed(self):
        (self.old/'unexpected').mkdir()
        self.reject()

    def test_unexpected_native_inventory_fails_closed(self):
        (self.native/'adoption.json').write_bytes(b'{}')
        self.reject(self.native_sid)

    def test_native_npz_and_ledger_hash_tampering_fail_closed(self):
        for name in ('arrays.npz','oov_ledger.jsonl.gz'):
            with self.subTest(file=name):
                path=self.native/name
                original=path.read_bytes()
                path.write_bytes(b'corrupt')
                self.reject(self.native_sid)
                path.write_bytes(original)

    def test_resigned_adoption_scenario_ordinal_source_science_proof_mismatch(self):
        path=self.adopted/'adoption.json'
        original=path.read_bytes()
        changes = {'scenario':lambda p:p['binding'].update(scenario_id=self.f.ids[1]),
            'ordinal':lambda p:p['binding'].update(archive_ordinal=2),
            'source':lambda p:p['binding'].update(source_archive={'foreign':True}),
            'science':lambda p:p['binding'].update(scientific_extractor_hashes={}),
            'proof':lambda p:p.update(successful_publication_proof_sha256='0'*64),
            'content':lambda p:p['v7_block_record'].update(array_content_sha256='0'*64)}
        for name,mutate in changes.items():
            with self.subTest(binding=name):
                self.change_json(path,lambda r:mutate(r['payload']),True)
                self.reject()
                path.write_bytes(original)

    def test_resigned_v7_source_scientific_member_scenario_bindings_fail_closed(self):
        path=self.old/'manifest.json'
        original=path.read_bytes()
        changes = {'source':lambda b:b.update(source_archive={'foreign':True}),
            'science':lambda b:b.update(scientific_extractor_hashes={}),
            'members':lambda b:b.update(source_member_set_sha256='0'*64),
            'scenario':lambda b:b.update(scenario_id=self.f.ids[1]),
            'ordinal':lambda b:b.update(archive_ordinal=2),
            'pseudo':lambda b:b.update(pseudo_rows=1),
            'partition':lambda b:b.update(partition='FIT_NORMAL'),
            'GNSS':lambda b:b.update(GNSS='included')}
        for name,mutate in changes.items():
            with self.subTest(binding=name):
                self.change_json(path,lambda r:mutate(r['payload']['binding']),True)
                self.reject()
                path.write_bytes(original)

    def test_fully_resigned_npz_provenance_dtype_tick_keys_and_feature_changes_fail_closed(self):
        saved={p.name:p.read_bytes() for p in self.old.iterdir()}
        mutations={'split':lambda a:a.update(source_split=np.array(['CAL_NORMAL']*2999)),
            'scenario':lambda a:a.update(scenario_id=np.array([self.f.ids[1]]*2999)),
            'town':lambda a:a.update(town=np.array(['Town02']*2999)),
            'tick':lambda a:a.update(tick=np.arange(2999,dtype=np.int64)),
            'tick_dtype':lambda a:a.update(tick=np.arange(1,3000,dtype=np.float64)),
            'feature_dtype':lambda a:a.update(camera=a['camera'].astype(np.float32)),
            'nonfinite':lambda a:a['camera'].__setitem__((0,0),np.nan),
            'extra_key':lambda a:a.update(gnss=np.ones((2999,4))),
            'missing_key':lambda a:a.pop('imu'),
            'changed_finite_feature':lambda a:a['camera'].__setitem__((0,0),123.456)}
        for name,mutate in mutations.items():
            with self.subTest(attack=name):
                arrays=data.load_npz(self.old/'arrays.npz')
                mutate(arrays)
                (self.old/'arrays.npz').write_bytes(data.npz_bytes(arrays))
                manifest=c.read_json(self.old/'arrays.json')
                manifest.update(npz_sha256=c.hash_file(self.old/'arrays.npz'),array_content_sha256=data.array_digest(arrays))
                (self.old/'arrays.json').write_bytes(c.canonical(manifest))
                def update(r):
                    r['payload'].update(npz_sha256=manifest['npz_sha256'],array_content_sha256=manifest['array_content_sha256'],
                        scientific_manifest_sha256=c.hash_file(self.old/'arrays.json'))
                self.change_json(self.old/'manifest.json',update,True)
                self.reject()
                for file,raw in saved.items():
                    (self.old/file).write_bytes(raw)

    def test_path_reparse_junction_flag_rejected_before_read(self):
        original=Path.lstat
        target=self.old
        def reparse(path,*args,**kwargs):
            value=original(path,*args,**kwargs)
            if path==target:
                return types.SimpleNamespace(st_mode=value.st_mode,
                    st_file_attributes=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',1024))
            return value
        with patch.object(Path,'lstat',reparse):
            with self.assertRaisesRegex(c.ReviewRequired,'Reparse path rejected'):
                self.load()

    def test_linked_resolution_path_rejected_before_read(self):
        original=Path.resolve
        def alias(path,*args,**kwargs):
            return self.f.base/'escaped' if path==self.old else original(path,*args,**kwargs)
        with patch.object(Path,'resolve',alias):
            with self.assertRaisesRegex(c.ReviewRequired,'Linked/junction path rejected'):
                self.load()

    def test_readback_change_between_verification_and_consumption_fails_closed(self):
        original=self.f.old.verify_block
        count=[0]
        def race(*args,**kwargs):
            result=original(*args,**kwargs)
            count[0]+=1
            if count[0]==2:
                path=self.old/'arrays.json'
                path.write_bytes(path.read_bytes()+b' ')
            return result
        with patch.object(self.f.old,'verify_block',race):
            self.reject()

    def test_incomplete_ambiguous_and_unknown_membership_fail_closed(self):
        self.reject('Town01/scenario-999')
        self.reject('../escape')
        self.f.ctx.expected[self.sid]['kind']='ambiguous'
        self.reject()

    def test_historical_proof_failure_or_checkpoint_tampering_rejected(self):
        path=self.f.old_store/'active.lock'
        path.write_bytes(b'changed')
        with self.assertRaises(Exception):
            self.f.adoption.audit_candidates(self.f.index,self.f.source,self.f.proof)


class AmendmentBindingTests(unittest.TestCase):
    def test_v1_failure_exact_seal_and_empty_units_verified(self):
        from gate2_forensics import verify_v1_failure
        evidence=verify_v1_failure()
        self.assertEqual(evidence,c.read_json(c.BUNDLE/'v1_failure_evidence.json'))
        self.assertEqual(evidence['committed_units'],{})
        self.assertFalse(evidence['scientific_result_obtained'])
        self.assertTrue(evidence['failure_preceded_agent_fit'])

    def test_ambiguous_import_origin_rejected(self):
        bogus=types.ModuleType('clean_block_store')
        bogus.__file__=str(SCRATCH/'unsealed.py')
        with patch.dict(sys.modules,clean_block_store=bogus):
            with self.assertRaises(c.ReviewRequired):
                resolution.authenticate_verifiers()

    def test_shadow_module_discovery_rejected_before_import_code_executes(self):
        original=resolution.importlib.util.find_spec
        def shadow(name,*args,**kwargs):
            if name=='v7_adoption':
                return types.SimpleNamespace(origin=str(SCRATCH/'shadow_adoption.py'))
            return original(name,*args,**kwargs)
        with (patch.object(resolution.importlib.util,'find_spec',shadow),
              patch.object(resolution.importlib,'import_module',side_effect=AssertionError('shadow must never execute')) as importer):
            with self.assertRaisesRegex(c.ReviewRequired,'Ambiguous sealed v8 discovery'):
                resolution.authenticate_verifiers()
            self.assertEqual(importer.call_count,0)

    def test_distinct_runtime_and_authorization_cannot_resume_v1(self):
        from execute_gate2 import authorization
        self.assertNotEqual(c.TARGET,c.V1_RUNTIME)
        self.assertEqual(c.TARGET.name,'gate2_train_health_v3')
        with self.assertRaises(c.ReviewRequired):
            authorization('HUMAN_REVIEWED_EXPERIMENT_2B_GATE2_TRAIN_HEALTH_V1')
        with self.assertRaises(c.ReviewRequired):
            authorization('HUMAN_REVIEWED_EXPERIMENT_2B_GATE2_TRAIN_HEALTH_V2')

    def test_future_runtime_binds_complete_resolutions_and_historical_failure(self):
        from gate2_runtime import manifest_for
        e={'bindings':{'prediction_semantics':c.SEMANTICS,'v1_failure_evidence_sha256':'a'*64,
                       'v2_failure_evidence_sha256':'c'*64},
           'environment':{},'source_binding':{},'membership':{},'completion':{},
           'block_resolutions':{'synthetic':True}}
        manifest=manifest_for('b'*64,e)
        self.assertEqual(manifest['block_resolutions'],e['block_resolutions'])
        self.assertEqual(manifest['historical_v1_failure_binding'],'a'*64)
        self.assertEqual(manifest['schema'],'transactional-Experiment-2B-Gate2-v3')
        self.assertEqual(manifest['historical_v2_failure_binding'],'c'*64)
        self.assertEqual(manifest['FIT_policy'],'Recompute frozen FIT under v3 manifest; no cross-runtime adoption')

    def test_scientific_implementation_is_byte_identical_to_v1(self):
        for name in ('gate2_science.py','gate2_audit.py','execute_gate2.py'):
            self.assertEqual((c.BUNDLE/name).read_bytes(),(c.V1/name).read_bytes())
        for name in ('gate2_data.py','gate2_resolution.py'):
            self.assertEqual((c.BUNDLE/name).read_bytes(),(c.V2/name).read_bytes())
