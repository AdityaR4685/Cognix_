"""Future-authorized FIT replay, using unchanged authenticated full TRAIN parser/inventory."""
import shutil
import time
import numpy as np
import graph_common
from graph_common import *
from graph_export import *
from current_upstream import authenticate_gate2
from current_restore import restore_current


class FITSink:
    def __init__(self,modules,science,agents,admission,records,replay_inventory,source_binding):
        require(graph_common.ACTIVE_PHASE=='graph-data','Real source sink forbidden during preparation')
        self.modules,self.science,self.agents,self.admission=modules,science,agents,admission
        self.records,self.replay_inventory=iter(records),replay_inventory
        self.source_binding=source_binding;self.ordinals_verified=[];self.summaries=[]
        self.current=None;self.stream=None;self.raw=None

    def scenario_begin(self,record):
        expected=next(self.records,None)
        require(expected is not None and expected['scenario_id']==record['scenario_id'] and
            expected['archive_ordinal']==record['archive_ordinal'] and
            expected['first_tar_offset']==record['first_tar_header_offset'],'TRAIN source order mismatch')
        self.record_sha=self.replay_inventory.bind(expected,len(self.ordinals_verified)+1)
        self.current=expected;self.expected_members=iter(expected['members'])
        self.needed=expected['scenario_id'] in self.admission.roles
        self.captured={};self.raw=None
        if self.needed:
            self.raw=PENDING/('raw-'+expected['scenario_id'].replace('/','__'))
            self.raw.mkdir()
            write_json(self.raw/'owned_workspace.json',dict(scenario_id=expected['scenario_id'],record_sha256=self.record_sha))

    def member_begin(self,record):
        self.stream=None;self.capture_path=None
        if record['scenario_id'] is None: return
        require(self.current is not None and record['scenario_id']==self.current['scenario_id'],'Unexpected member scenario')
        if not self.needed or record['type']=='5': return
        relative='/'.join(record['path'].split('/')[3:])
        wanted=relative=='imu.feather'
        if relative.startswith(('rgb-front/','segmentation-front/')):
            tick=int(Path(relative).stem)
            if 1<=tick<=2999:
                m=modality_for_recipe(recipe_for_tick(tick))
                wanted=(m=='Camera' and relative.startswith('rgb-front/')) or (m=='Seg' and relative.startswith('segmentation-front/'))
        if wanted:
            require(record['size']<=128<<20 and shutil.disk_usage(self.raw).free>=(20<<30)+record['size'],
                    'Insufficient bounded one-scenario raw scratch reserve')
            self.capture_path=safe_path(self.raw/relative)
            require(self.capture_path.is_relative_to(self.raw),'Raw member escape')
            self.capture_path.parent.mkdir(parents=True,exist_ok=True)
            self.stream=self.capture_path.open('xb')

    def member_data(self,data):
        if self.stream: require(self.stream.write(data)==len(data),'Short raw member write')

    def member_complete(self,record):
        if self.stream:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
            require(self.capture_path.stat().st_size==record['size'] and hash_file(self.capture_path)==record['sha256'],
                    'FIT raw member size/hash mismatch')
            self.captured[record['path']]=record
        if record['scenario_id'] is not None:
            require(next(self.expected_members,None)==record,'Exact sealed TRAIN member disagreement')

    def scenario_complete(self,record):
        require(self.replay_inventory.bind(self.current,len(self.ordinals_verified)+1)==self.record_sha and
            record['scenario_id']==self.current['scenario_id'] and next(self.expected_members,None) is None and
            record['member_chain_sha256']==self.current['parser_member_chain_sha256'],
            'Incomplete/mismatched scenario or changed inventory binding')
        self.ordinals_verified.append(record['archive_ordinal'])
        if not self.needed: return dict(CAL_materialized=False)
        sid=self.current['scenario_id']
        clean=self.modules['gate2_data'].load_clean(sid)
        # Canonical resolver authenticates manifest/provenance, not a caller-supplied cache path.
        require(clean['source_split']=='train' and clean['role']=='clean','Clean parent provenance failed')
        import pandas as pd
        from PIL import Image
        imu=pd.read_feather(self.raw/'imu.feather')[list(self.science['IMU_ACCEL_COLUMNS'])].to_numpy(dtype=np.float64)
        require(imu.shape==(3000,3) and np.isfinite(imu).all(),'Raw IMU schema/domain')
        rows=[];ledgers=[]
        for tick in range(1,3000):
            m=modality_for_recipe(recipe_for_tick(tick))
            relative=f'rgb-front/{tick:06d}.jpg' if m=='Camera' else f'segmentation-front/{tick:06d}.png' if m=='Seg' else 'imu.feather'
            member=self.captured['train/'+sid+'/'+relative]
            if m=='IMU': payload=causal_window(imu,tick)
            else:
                with Image.open(self.raw/relative) as image: payload=np.asarray(image)
            parent={mod:clean['arrays'][mod.lower()][tick-1] for mod in NODE_ORDER}
            pair,ledger=generate_pair(self.science,self.agents,self.admission,sid,tick,payload,parent)
            ledger.update(source_member={k:member[k] for k in ('path','size','sha256')},
                source_archive_sha256=SOURCE['sha256'],scenario_inventory_record_sha256=self.record_sha,
                parent_clean_block_sha256=clean['block_npz_sha256'],
                parent_clean_block_content_sha256=clean['block_content_sha256'])
            rows.extend(pair);ledgers.append(ledger)
        binding=dict(source_identity=SOURCE,Gate2_runtime_seal=SEALS['gate2_train_health_v3'],
            graph_partition_sha256=GRAPH_PARTITION_SHA,graph_partition_seal=SEALS['graph_partition_freeze_v1'],
            protocol_scientific_sha256=PROTOCOL_SHA,canonical_clean_resolution=clean['resolved_provenance'],
            parent_clean_block_sha256=clean['block_npz_sha256'],scenario_inventory_record_sha256=self.record_sha)
        summary=write_scenario(PENDING/'units'/sid.replace('/','__'),sid,rows,ledgers,self.admission,binding)
        self.summaries.append(summary)
        owned=safe_path(self.raw)
        require(owned.is_relative_to(PENDING) and read_json(owned/'owned_workspace.json')==
            dict(scenario_id=sid,record_sha256=self.record_sha),'Unsafe scratch cleanup')
        for item in owned.rglob('*'): safe_path(item)
        shutil.rmtree(owned);self.raw=None
        return dict(FIT_graph_pairs=len(rows)//2,CAL_materialized=False)

    def close(self):
        if self.stream: self.stream.close();self.stream=None


def replay_current():
    require(graph_common.ACTIVE_PHASE=='graph-data','Future replay forbidden in preparation')
    modules=authenticate_gate2();modules['gate2_resolution'].real_context()
    science=modules['gate2_science'].load_science();agents=restore_current(science)
    admission=Admission(read_json(BUNDLE/'current_partition_audit.json'))
    receipt=read_json(REPORT/'gate1_execution_bundle_v8/source_verification.json')
    source_binding={k:receipt[k] for k in ('fingerprint','identity','independently_measured_full_sha256','schema')}
    replay_inventory=modules['gate2_replay_inventory'].load_verified_replay_inventory(source_binding)
    import local_inventory
    from train_replay import TrainReplay
    records=local_inventory.inventory_records(REPORT/'gate1_execution_bundle_v8',replay_inventory.index)
    sink=FITSink(modules,science,agents,admission,records,replay_inventory,source_binding)
    parser=TrainReplay(sink=sink,on_member=lambda member:None)
    fingerprint=source_fingerprint();last=time.monotonic()
    try:
        with Path(SOURCE['path']).open('rb') as stream:
            for raw in iter(lambda:stream.read(1<<20),b''):
                parser.feed(raw)
                if time.monotonic()-last>=25:
                    print('Future FIT-only replay: %d compressed bytes; %d scenarios verified' %
                          (parser.compressed_bytes,len(sink.ordinals_verified)),flush=True);last=time.monotonic()
        require(parser.finished and parser.decoder.eof and parser.tar_end and parser.compressed_bytes==SOURCE['bytes'] and
            parser.compressed_hash.hexdigest()==SOURCE['sha256'] and sink.ordinals_verified==list(range(1,102)) and
            next(sink.records,None) is None and source_fingerprint()==fingerprint,'Incomplete full source replay')
        replay_inventory.finish(sink.ordinals_verified)
        require(set(s['scenario_id'] for s in sink.summaries)==set(admission.partition['FIT_NORMAL']) and
                len(sink.summaries)==76,'Incomplete FIT graph export')
        return dict(status='TRAIN_ONLY_GRAPH_DATA_COMPLETE',source_identity=SOURCE,full_source_replay_verified=True,
            all_members_exactly_matched=True,gzip_eof=True,TAR_end=True,ordinals_verified=sink.ordinals_verified,
            CAL_raw_materialized=False,GNSS_raw_materialized=False,clean_features_recomputed=False,
            upstream_refit=False,recalibration=False,TEST_requests=0,network_requests=0,
            unit_summaries=sorted(sink.summaries,key=lambda s:s['scenario_id']),
            scientific_content_sha256=digest(canonical([{k:s[k] for k in
                ('scenario_id','graph_role','scientific_content_sha256')} for s in
                sorted(sink.summaries,key=lambda s:s['scenario_id'])])),
            graph_partition_sha256=GRAPH_PARTITION_SHA,Gate2_runtime_seal=SEALS['gate2_train_health_v3'],
            limitation='Development-only: all 76 FIT, including 15 GRAPH_VAL, fitted upstream representation')
    finally: sink.close()
