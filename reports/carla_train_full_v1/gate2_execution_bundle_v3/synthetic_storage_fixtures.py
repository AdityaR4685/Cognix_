"""Artificial 3000-tick evidence for exact native/adoption verifier contracts."""
from pathlib import Path
import numpy as np
from gate2_common import canonical, digest, hash_file, write_new
from gate2_data import npz_bytes, array_digest
from gate2_resolution import ResolutionContext


class StorageFixture:
    def clone(self, base):
        """Fresh artificial evidence per attack; immutable pristine fixture reused."""
        import copy
        import shutil
        result = object.__new__(StorageFixture)
        result.base, result.native, result.old, result.adoption = base, self.native, self.old, self.adoption
        result.store, result.old_store = base/'v8', base/'v7'
        shutil.copytree(self.store, result.store)
        shutil.copytree(self.old_store, result.old_store)
        result.ids, result.features = list(self.ids), self.features
        for name in ('source','science','old_science','membership','scenarios','index','proof'):
            setattr(result,name,copy.deepcopy(getattr(self,name)))
        result.proof['v7_runtime_root'] = str(result.old_store)
        result.paths = {sid:(result.old_store if i<2 else result.store)/'verified_clean_blocks'/
                        self.native.block_name(sid) for i,sid in enumerate(result.ids)}
        expected = [copy.deepcopy(self.ctx.expected[sid]) for sid in result.ids]
        for i,sid in enumerate(result.ids[:2]):
            path = result.store/'verified_clean_blocks'/self.native.block_name(sid)
            payload = self.adoption.adoption_payload(result.scenarios[i],result.source,result.science,
                result.proof,result.proof['completed_blocks'][i],expected[i]['oov_ledger'])
            (path/'adoption.json').write_bytes(canonical({'schema':'immutable-v7-to-v8-adoption',
                'payload':payload,'payload_sha256':self.native.digest(payload)}))
            expected[i].update(manifest_sha256=hash_file(path/'adoption.json'),
                               v7_block_path=str(result.paths[sid]))
        result.ctx = ResolutionContext(result.store,result.old_store,result.scenarios,result.source,
            result.science,result.proof,expected,(result.native,result.old,result.adoption),result.membership)
        return result

    def __init__(self, base):
        import clean_block_store as native
        import v7_clean_block_store as old
        import v7_adoption as adoption
        import segmentation_oov as oov
        from atomic_publication import publish_state
        self.base, self.native, self.old, self.adoption = base, native, old, adoption
        self.store, self.old_store = base / 'v8', base / 'v7'
        self.source = {'schema': 'synthetic-source-only', 'identity': {'synthetic': True}}
        self.science, self.old_science = native.scientific_hashes(), old.scientific_hashes()
        self.ids = ['Town01/scenario-' + str(i) for i in range(1, 5)]
        self.membership = {'FIT_NORMAL': [self.ids[0], self.ids[2]],
                           'CAL_NORMAL': [self.ids[1], self.ids[3]]}
        self.scenarios, self.paths, self.features = [], {}, {}
        for i, sid in enumerate(self.ids, 1):
            members = [{'path': 'train/' + sid + f'/segmentation-front/{t:06d}.png',
                        'size': 1, 'sha256': digest(f'artificial-{sid}-{t}'.encode())}
                       for t in range(3000)]
            self.scenarios.append({'scenario_id': sid, 'archive_ordinal': i, 'town': 'Town01',
                'n_ticks': 3000, 'emitted_rows': 2999,
                'source_member_set_sha256': native.digest(sorted(members, key=lambda r:r['path']))})
            self.features[sid] = self.arrays(sid, i)
            root = self.old_store if i <= 2 else self.store
            path = root / 'verified_clean_blocks' / native.block_name(sid)
            path.mkdir(parents=True)
            self.paths[sid] = path
            write_new(path / 'arrays.npz', npz_bytes(self.features[sid]))
            science = self.old_science if i <= 2 else self.science
            binding = (old if i <= 2 else native).block_binding(self.scenarios[-1], self.source, science)
            manifest = {'scenario_id': sid, 'n_raw_ticks': 3000, 'n_clean_rows': 2999,
                'first_clean_tick': 1, 'window_ticks': 12, 'active_dimensions': {'camera':18,'seg':29,'imu':10},
                'partition': None, 'pseudo_rows': 0, 'fitting_executed': False,
                'anomaly_observation_expected': False, 'prediction_semantics': binding['prediction_semantics'],
                'GNSS': 'validated metadata only; excluded from arrays and downstream',
                'clean_role': 'official TRAIN source role; preregistered constructed normal observation',
                'scientific_source_bindings': science['source_files'], 'dependencies': science['dependencies'],
                'train_contract_sha256': binding['train_contract_sha256'],
                'npz_sha256': hash_file(path / 'arrays.npz'),
                'array_content_sha256': array_digest(self.features[sid]), 'source_members': members}
            write_new(path / 'arrays.json', canonical(manifest))
            payload = {'binding': binding, 'npz_sha256': manifest['npz_sha256'],
                'array_content_sha256': manifest['array_content_sha256'],
                'scientific_manifest_sha256': hash_file(path / 'arrays.json')}
            if i > 2:
                write_new(path / 'oov_ledger.jsonl.gz', self.ledger(sid, members, False, oov))
                payload['oov_ledger'] = oov.validate_ledger(path / 'oov_ledger.jsonl.gz', self.scenarios[-1], members)
            write_new(path / 'manifest.json', canonical({'schema': 'immutable-clean-block-v7' if i <= 2 else
                'immutable-clean-block-v8', 'payload': payload, 'payload_sha256': native.digest(payload)}))
        old_records = [old.verify_block(self.paths[sid], self.scenarios[i], self.source, self.old_science)
                       for i, sid in enumerate(self.ids[:2])]
        self.index = {'schema': 'frozen-local-TRAIN-inventory-v7', 'ledger_sha256': 'a'*64,
            'ledger_logical_sha256': 'b'*64, 'scenario_index_sha256': 'c'*64, 'scenario_count': 4,
            'total_tar_members': 24000, 'train_contract_sha256': hash_file(native.BUNDLE / 'train_contract.json'),
            'scenarios': self.scenarios}
        inv = {k:self.index[k] for k in ('schema','ledger_sha256','ledger_logical_sha256',
               'scenario_index_sha256','scenario_count','total_tar_members','train_contract_sha256')}
        state = {'schema':'local-scenario-progress-v7','source_binding':self.source,'status':'EXTRACTING',
            'gate1_complete':False,'inventory_binding':inv,'partition':None,'pseudo_rows':0,
            'fitting_executed':False,'GAT_executed':False,'TEST_requests':0,'TRAIN_network_requests':0,
            'network_requests':0,'scientific_extractor_hashes':self.old_science,
            'completed_blocks':old_records,'completed_scenarios':2}
        publish_state(self.old_store / 'state.json', state)
        write_new(self.old_store / 'active.lock', b'synthetic-preserved-lock')
        failure = self.old_store / 'runs' / 'synthetic_failure' / 'failure.json'
        failure.parent.mkdir(parents=True)
        write_new(failure, canonical({'exception':'synthetic OOV failure','completed_blocks':old_records}))
        self.proof = {'schema':'sealed-v7-success-prefix-and-failure-proof-v8',
            'v7_runtime_root':str(self.old_store), 'v7_seal_sha256':adoption.V7_SEAL,
            'source_binding':self.source,'v7_scientific_hashes':self.old_science,
            'completed_scenarios':2,'completed_blocks':old_records,
            'first_uncompleted_scenario':self.ids[2],'failure_exception':'synthetic OOV failure',
            'publication_evidence':[{'relative_path':p.relative_to(self.old_store).as_posix(),
                'sha256':hash_file(p),'kind':kind} for p,kind in
                ((self.old_store/'state.json','successful_publication_checkpoint'),
                 (self.old_store/'active.lock','preserved_failure_lock'), (failure,'invocation_failure'))]}
        adoption.audit_candidates(self.index, self.source, self.proof)
        expected = []
        for i, sid in enumerate(self.ids):
            scenario = self.scenarios[i]
            if i < 2:
                path = self.store / 'verified_clean_blocks' / native.block_name(sid)
                path.mkdir(parents=True)
                members = __import__('gate2_common').read_json(self.paths[sid] / 'arrays.json')['source_members']
                write_new(path / 'oov_ledger.jsonl.gz', self.ledger(sid, members, True, oov))
                ledger = oov.validate_ledger(path/'oov_ledger.jsonl.gz',scenario,members,adopted=True)
                payload = adoption.adoption_payload(scenario,self.source,self.science,self.proof,old_records[i],ledger)
                write_new(path/'adoption.json',canonical({'schema':'immutable-v7-to-v8-adoption',
                    'payload':payload,'payload_sha256':native.digest(payload)}))
                expected.append(adoption.verify_adoption(path,scenario,self.source,self.science,self.proof))
            else:
                expected.append(native.verify_block(self.paths[sid],scenario,self.source,self.science))
        self.ctx = ResolutionContext(self.store,self.old_store,self.scenarios,self.source,self.science,
            self.proof,expected,(native,old,adoption),self.membership)

    @staticmethod
    def arrays(sid, marker):
        return {'camera':np.full((2999,18), marker + .125, dtype=np.float64),
                'seg':np.full((2999,29), marker / 29, dtype=np.float64),
                'imu':np.full((2999,10), marker + .25, dtype=np.float64),
                'tick':np.arange(1,3000,dtype=np.int64), 'scenario_id':np.array([sid]*2999),
                'source_split':np.array(['train']*2999), 'town':np.array(['Town01']*2999)}

    @staticmethod
    def ledger(sid, members, adopted, oov):
        # Exact frame schema and policy, synthetic member bytes only. The frozen
        # validator is called independently after encoding all 3000 ticks.
        policy = oov.policy_digest()
        records = [{'schema':'Experiment-2B-OOV-frame-v8','scenario_id':sid,
            'frame':f'{t:06d}.png','tick':t,'total_semantic_pixels':None if adopted else 64,
            'count_remapped':0,'unique_oov_ids':[],'per_id_counts':{},'source_png_member':member,
            'preprocessing_policy_sha256':policy,'evidence':'V7_SUCCESS_ACCEPTED_DOMAIN_PROOF' if adopted
            else 'DIRECT_PREPROCESSING_OBSERVATION','descriptive_only':True,'decision_use':'none'}
            for t,member in enumerate(members)]
        return oov.encode_ledger(records)
