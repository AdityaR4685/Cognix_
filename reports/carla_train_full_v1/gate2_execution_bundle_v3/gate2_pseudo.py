"""Future-only CAL raw replay and matched pseudo features; frozen recipes and seeds.

The complete unchanged TRAIN parser validates every TAR member. Only raw CAL
payloads needed by active scheduled recipes are materialized in new owned
scratch. FIT raw payloads and GNSS payloads are never materialized or corrupted.
"""
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from gate2_common import (V8, SOURCE, MODALITIES, SEVERITIES, require, canonical,
                         write_new, hash_file, read_json, safe_path, pretty)
from gate2_data import (load_clean, recipe_for_tick, validate_cal_row, npz_bytes, array_digest)


def generate_tick(science, membership, sid, tick, payload, parent):
    require(sid in membership['CAL_NORMAL'] and sid not in membership['FIT_NORMAL'],
            'Pseudo parent is outside CAL')
    rid = recipe_for_tick(tick)
    if rid is None:
        return None, None
    modality = 'Camera' if rid.startswith('camera') else 'Seg' if rid.startswith('seg') else 'IMU'
    baseline = np.asarray(payload)
    preprocessing = None
    if modality == 'Seg':
        # Exactly the upstream-bound Experiment-2B amendment, before the recipe.
        baseline, preprocessing = science['sanitize'](baseline)
    fn = science[{'Camera': 'camera_embedding_features', 'Seg': 'segmentation_histogram_features',
                  'IMU': 'imu_window_features'}[modality]]
    clean = fn(baseline)
    require(np.array_equal(clean, parent[modality.lower()]), 'Raw pseudo parent disagrees with sealed clean block')
    sample = science['generate_pseudo_anomaly']({modality: baseline}, recipe_id=rid,
        source_split=science['CarlAnomalySplit'].TRAIN, source_scenario=sid,
        source_tick=tick, seed=tick, severity=SEVERITIES[rid])
    features = fn(np.asarray(sample.data))
    require(features.shape == parent[modality.lower()].shape and np.isfinite(features).all(),
            'Pseudo feature finite/dimension failure')
    if np.allclose(features, parent[modality.lower()], rtol=1e-12, atol=1e-12):
        return None, {'recipe_id': rid, 'reason': 'no_effect', 'parent_tick': tick}
    row = sample.provenance()
    row.update(partition='CAL_NORMAL', parent_scenario=sid, parent_tick=tick,
               window_start_tick=max(0, tick - 11), window_end_tick=tick,
               target_normal=0, graph_role=None, upstream_fit=False,
               preprocessing_policy_applied_before_recipe=modality == 'Seg',
               segmentation_preprocessing_counts=preprocessing,
               parent_feature_sha256=array_digest({modality.lower(): parent[modality.lower()]}),
               pseudo_feature_sha256=array_digest({modality.lower(): features}))
    validate_cal_row(row, membership, modality)
    combined = {m.lower(): parent[m.lower()].copy() for m in MODALITIES}
    combined[modality.lower()] = features
    return combined, row


def write_jsonl_gzip(path, rows):
    import io
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write(canonical(row) + b'\n')
    write_new(path, buffer.getvalue())


def read_pseudo(unit):
    from gate2_data import load_npz
    with gzip.open(unit / 'provenance.jsonl.gz', 'rb') as stream:
        rows = [json.loads(line) for line in stream]
    return load_npz(unit / 'pseudo_features.npz'), rows


class CALSink:
    def __init__(self, runtime, science, records, source_binding, replay_inventory):
        self.runtime, self.science, self.membership = runtime, science, runtime.evidence['membership']
        self.records, self.source_binding = iter(records), source_binding
        self.replay_inventory = replay_inventory
        self.scenario_inventory_record_sha256 = None
        self.current = None
        self.stream = None
        self.ordinals_verified = []
        self.completed_at_start = runtime.verify()['committed']

    def scenario_begin(self, record):
        expected = next(self.records, None)
        require(expected is not None and expected['scenario_id'] == record['scenario_id'] and
                expected['archive_ordinal'] == record['archive_ordinal'] and
                expected['first_tar_offset'] == record['first_tar_header_offset'], 'TRAIN source order mismatch')
        self.scenario_inventory_record_sha256 = self.replay_inventory.bind(expected,len(self.ordinals_verified)+1)
        self.current = expected
        self.expected_members = iter(expected['members'])
        sid = expected['scenario_id']
        self.unit_name = 'cal-' + sid.replace('/', '__')
        self.needed = sid in self.membership['CAL_NORMAL'] and self.unit_name not in self.completed_at_start
        self.raw = None
        self.captured = {}
        if self.needed:
            self.raw = self.runtime.attempt / ('raw-' + sid.replace('/', '__'))
            self.raw.mkdir()
            write_new(self.raw / 'owned_workspace.json', canonical({'scenario_id': sid,
                'run_manifest_sha256': __import__('gate2_common').digest(canonical(self.runtime.manifest))}))

    def member_begin(self, record):
        self.stream = None
        self.capture_path = None
        if record['scenario_id'] is None:
            return
        require(self.current and record['scenario_id'] == self.current['scenario_id'], 'Unexpected source member')
        if not self.needed or record['type'] == '5':
            return
        relative = '/'.join(record['path'].split('/')[3:])
        wanted = relative == 'imu.feather'
        if relative.startswith(('rgb-front/', 'segmentation-front/')):
            tick = int(Path(relative).stem)
            if tick >= 1:
                rid = recipe_for_tick(tick)
                wanted = rid is not None and ((rid.startswith('camera') and relative.startswith('rgb-front/')) or
                                               (rid.startswith('seg') and relative.startswith('segmentation-front/')))
        if wanted:
            require(record['size'] <= 128 << 20 and shutil.disk_usage(self.raw).free >= (20 << 30) + record['size'],
                    'Insufficient bounded raw CAL scratch reserve')
            self.capture_path = self.raw / relative
            safe_path(self.capture_path)
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.capture_path.open('xb')

    def member_data(self, data):
        if self.stream:
            require(self.stream.write(data) == len(data), 'Short CAL raw write')

    def member_complete(self, record):
        if self.stream:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None
            require(self.capture_path.stat().st_size == record['size'] and
                    hash_file(self.capture_path) == record['sha256'], 'CAL raw member hash mismatch')
            self.captured[record['path']] = record
        if record['scenario_id'] is not None:
            expected = next(self.expected_members, None)
            require(expected == record, 'Exact sealed TRAIN member disagreement')

    def scenario_complete(self, record):
        require(self.replay_inventory.bind(self.current,len(self.ordinals_verified)+1) ==
                self.scenario_inventory_record_sha256, 'Replay record changed before CAL consumption')
        require(self.current['scenario_id'] == record['scenario_id'] and
                next(self.expected_members, None) is None and
                record['member_chain_sha256'] == self.current['parser_member_chain_sha256'],
                'Incomplete/mismatched CAL source scenario')
        self.ordinals_verified.append(record['archive_ordinal'])
        if not self.needed:
            return {'active_CAL_materialized': False}
        sid = self.current['scenario_id']
        clean = load_clean(sid)
        import pandas as pd
        import PIL.Image
        df = pd.read_feather(self.raw / 'imu.feather')
        imu = df[list(self.science['IMU_ACCEL_COLUMNS'])].to_numpy(dtype=np.float64)
        require(imu.shape == (3000, 3) and np.isfinite(imu).all(), 'Raw CAL IMU domain failure')
        vectors, rows, skipped, attempts = [], [], [], {}
        for tick in range(1, 3000):
            rid = recipe_for_tick(tick)
            if rid is None:
                continue
            attempts[rid] = attempts.get(rid, 0) + 1
            if rid.startswith('camera'):
                relative = f'rgb-front/{tick:06d}.jpg'
            elif rid.startswith('seg'):
                relative = f'segmentation-front/{tick:06d}.png'
            else:
                relative = 'imu.feather'
            source_member = self.captured['train/' + sid + '/' + relative]
            if relative == 'imu.feather':
                payload = imu[max(0, tick - 11):tick + 1]
            else:
                with PIL.Image.open(self.raw / relative) as image:
                    payload = np.asarray(image)
            parent = {m.lower(): clean['arrays'][m.lower()][tick - 1] for m in MODALITIES}
            combined, row = generate_tick(self.science, self.membership, sid, tick, payload, parent)
            if combined is None:
                skipped.append(row)
                continue
            row.update(source_member={key: source_member[key] for key in ('path', 'size', 'sha256')},
                parent_clean_block_sha256=clean['block_npz_sha256'],
                preprocessing_policy_sha256=self.runtime.evidence['bindings']['preprocessing_policy_sha256'],
                source_archive_sha256=SOURCE['sha256'])
            vectors.append(combined)
            rows.append(row)
        arrays = {m.lower(): np.array([v[m.lower()] for v in vectors], dtype=np.float64).reshape(-1, dim)
                  for m, dim in __import__('gate2_common').DIMS.items()}
        stats = {'scenario_id': sid, 'source_split': 'train', 'partition': 'CAL_NORMAL',
                 'attempts_by_recipe': attempts, 'skipped_no_effect': skipped,
                 'GNSS_rotation_slots_skipped': sum(recipe_for_tick(t) is None for t in range(1, 3000)),
                 'emitted_pseudo_rows': len(rows), 'pseudo_array_content_sha256': array_digest(arrays),
                 'max_pseudo_per_tick': 1, 'seed_rule': '0+tick', 'upstream_fit': False, 'graph_role': None}
        def writer(pending):
            write_new(pending / 'pseudo_features.npz', npz_bytes(arrays))
            write_jsonl_gzip(pending / 'provenance.jsonl.gz', rows)
            write_new(pending / 'pseudo_summary.json', pretty(stats))
        self.runtime.commit(self.unit_name, writer, {'source_binding': self.source_binding,
            'scenario_inventory_record_sha256': self.scenario_inventory_record_sha256,
            'clean_block_sha256': clean['block_npz_sha256']})
        # Only owned successful scratch is released, after sealed unit readback.
        owned = safe_path(self.raw)
        require(owned.is_relative_to(self.runtime.attempt) and owned.name == 'raw-' + sid.replace('/', '__'),
                'Unsafe CAL scratch cleanup')
        for item in owned.rglob('*'):
            safe_path(item)
        shutil.rmtree(owned)
        return {'active_CAL_materialized': True, 'pseudo_rows': len(rows)}

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None


def prepare_cal_pseudo(runtime, science):
    if runtime.has('cal-source-verification'):
        require(all(runtime.has('cal-' + sid.replace('/', '__')) for sid in runtime.evidence['membership']['CAL_NORMAL']),
                'CAL source proof exists without every CAL unit')
        return
    import local_inventory
    from train_replay import TrainReplay
    from local_source import fingerprint
    binding = runtime.evidence['source_binding']
    require(fingerprint(SOURCE['path']) == binding['fingerprint'], 'Source fingerprint changed before CAL replay')
    from gate2_replay_inventory import load_verified_replay_inventory
    replay_inventory = load_verified_replay_inventory(binding)
    sink = CALSink(runtime, science, local_inventory.inventory_records(V8,replay_inventory.index), binding, replay_inventory)
    parser = TrainReplay(sink=sink, on_member=lambda member: None)
    last = time.monotonic()
    try:
        with Path(SOURCE['path']).open('rb') as stream:
            for raw in iter(lambda: stream.read(1 << 20), b''):
                parser.feed(raw)
                if time.monotonic() - last >= 30:
                    print('Future CAL-only replay: %s compressed bytes, %s scenarios verified' %
                          (parser.compressed_bytes, len(sink.ordinals_verified)), flush=True)
                    last = time.monotonic()
        require(parser.finished and parser.decoder.eof and parser.tar_end and
                parser.compressed_bytes == SOURCE['bytes'] and
                parser.compressed_hash.hexdigest() == SOURCE['sha256'] and
                sink.ordinals_verified == list(range(1, 102)) and
                next(sink.records, None) is None and fingerprint(SOURCE['path']) == binding['fingerprint'],
                'Incomplete or mismatched full CAL source replay')
        replay_inventory.finish(sink.ordinals_verified)
        def writer(pending):
            write_new(pending / 'source_replay.json', pretty({'compressed_bytes': parser.compressed_bytes,
                'compressed_sha256': parser.compressed_hash.hexdigest(), 'gzip_eof': True, 'TAR_end': True,
                'ordinals_verified': sink.ordinals_verified, 'all_members_exactly_matched': True,
                'FIT_raw_materialized': False, 'GNSS_raw_materialized': False, 'TEST_requests': 0,
                'network_requests': 0, 'graph_constructed': False}))
        runtime.commit('cal-source-verification', writer, {'source_binding': binding})
    finally:
        sink.close()
