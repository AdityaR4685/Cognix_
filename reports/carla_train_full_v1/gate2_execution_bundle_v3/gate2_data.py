"""Explicit FIT-clean and CAL-only roles; no implicit partition or cache rebuilding."""
import io
import numpy as np
from gate2_common import (STORE, MODALITIES, DIMS, RECIPES, SEVERITIES, require,
                          read_json, hash_file, digest)
from gate2_science import check_features


def array_digest(arrays):
    import hashlib
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_clean(sid, admitted=None):
    from gate2_resolution import resolve_arrays
    arrays, manifest, provenance = resolve_arrays(sid, admitted)
    require(manifest['scenario_id'] == sid and manifest['window_ticks'] == 12 and
            manifest['n_clean_rows'] == 2999 and manifest['n_raw_ticks'] == 3000 and
            manifest['first_clean_tick'] == 1 and manifest['partition'] is None and
            manifest['pseudo_rows'] == 0 and manifest['fitting_executed'] is False,
            'Clean-block role/window binding mismatch')
    require(set(arrays) == {'camera', 'seg', 'imu', 'tick', 'scenario_id', 'source_split', 'town'} and
            array_digest(arrays) == manifest['array_content_sha256'], 'Clean content/schema mismatch')
    require(np.array_equal(arrays['tick'], np.arange(1, 3000)) and
            np.all(arrays['scenario_id'] == sid) and np.all(arrays['source_split'] == 'train'),
            'Clean scenario/tick/split mismatch')
    for modality in MODALITIES:
        check_features(arrays[modality.lower()], modality)
    return {'scenario_id': sid, 'source_split': 'train', 'role': 'clean',
            'synthetic_corruption': False, 'arrays': arrays,
            'block_npz_sha256': manifest['npz_sha256'],
            'block_content_sha256': manifest['array_content_sha256'],
            'resolved_provenance': provenance}


def assemble_fit(membership, modality, reader=load_clean, expected_rows=2999):
    require(modality in MODALITIES and set(membership) == {'FIT_NORMAL', 'CAL_NORMAL'} and
            not set(membership['FIT_NORMAL']) & set(membership['CAL_NORMAL']), 'Invalid FIT roles')
    ids = sorted(membership['FIT_NORMAL'])
    require(ids and len(ids) == len(set(ids)), 'Empty/duplicate FIT membership')
    rows, bindings = [], []
    for sid in ids:
        record = reader(sid)
        require(record['scenario_id'] == sid and record['source_split'] == 'train' and
                record['role'] == 'clean' and record['synthetic_corruption'] is False,
                'CAL/pseudo/TEST input to upstream fitting')
        arrays = record['arrays']
        require(np.all(arrays['scenario_id'] == sid) and np.all(arrays['source_split'] == 'train') and
                len(arrays['tick']) == expected_rows and
                np.array_equal(arrays['tick'], np.arange(1, expected_rows + 1)), 'FIT row provenance mismatch')
        x = arrays[modality.lower()]
        check_features(x, modality)
        require(len(x) == expected_rows, 'FIT rows missing')
        rows.append(x)
        bindings.append({key: record[key] for key in
                         ('scenario_id', 'block_npz_sha256', 'block_content_sha256')})
        if 'resolved_provenance' in record:
            bindings[-1]['resolved_provenance'] = record['resolved_provenance']
    x = np.concatenate(rows)
    return x, {'FIT_NORMAL': membership['FIT_NORMAL'], 'CAL_NORMAL_excluded': membership['CAL_NORMAL'],
               'ordered_consumed_scenarios': ids, 'rows_per_scenario': expected_rows,
               'total_rows': len(x), 'input_array_sha256': array_digest({'features': x}),
               'clean_block_bindings': bindings, 'pseudo_rows': 0, 'TEST_rows': 0,
               'source_split': 'train', 'bootstrap_members': 5, 'bootstrap_seed': 42,
               'window_ticks': 12, 'dimensions': DIMS[modality]}


def recipe_for_tick(tick):
    require(type(tick) is int and tick >= 1, 'Invalid pseudo parent tick')
    rid = RECIPES[tick % len(RECIPES)]
    return None if rid.startswith('gnss') else rid


def validate_cal_row(row, membership, modality=None):
    require(row['source_split'] == 'train' and row['partition'] == 'CAL_NORMAL' and
            row['parent_scenario'] in membership['CAL_NORMAL'] and
            row['parent_scenario'] not in membership['FIT_NORMAL'], 'CAL pseudo parent isolation failure')
    tick = row['parent_tick']
    require(type(tick) is int and 1 <= tick <= 2999 and
            row['window_start_tick'] == max(0, tick - 11) and
            row['window_end_tick'] == tick, 'CAL causal window failure')
    if row['target_normal'] == 1:
        require(row['recipe_id'] is None and row['synthetic_corruption'] is False, 'Clean CAL class failure')
    else:
        require(row['target_normal'] == 0 and row['synthetic_corruption'] is True and
                row['recipe_id'] in SEVERITIES and row['recipe_id'] == recipe_for_tick(tick) and
                row['severity'] == SEVERITIES[row['recipe_id']] and row['seed'] == tick and
                row['modality'] in ('camera', 'seg', 'imu'), 'Pseudo recipe/severity/seed mismatch')
        require(modality is None or row['modality'] == modality.lower(), 'Unmatched pseudo channel')
    require(row['graph_role'] is None and row['upstream_fit'] is False, 'FIT graph pseudo forbidden')


def clean_cal_rows(sid, ticks):
    return [{'source_split': 'train', 'partition': 'CAL_NORMAL', 'parent_scenario': sid,
             'parent_tick': int(t), 'window_start_tick': max(0, int(t) - 11), 'window_end_tick': int(t),
             'target_normal': 1, 'recipe_id': None, 'synthetic_corruption': False,
             'graph_role': None, 'upstream_fit': False} for t in ticks]


def modality_calibration(membership, modality, pseudo_reader, clean_reader=load_clean):
    features, labels, provenance = [], [], []
    for sid in sorted(membership['CAL_NORMAL']):
        clean = clean_reader(sid)
        require(clean['scenario_id'] == sid and clean['role'] == 'clean' and
                clean['source_split'] == 'train' and not clean['synthetic_corruption'], 'CAL clean binding failure')
        x = clean['arrays'][modality.lower()]
        check_features(x, modality)
        features.append(x)
        labels.extend([1] * len(x))
        provenance.extend(clean_cal_rows(sid, clean['arrays']['tick']))
        arrays, rows = pseudo_reader(sid)
        used_ticks = set()
        for i, row in enumerate(rows):
            validate_cal_row(row, membership)
            require(row['parent_tick'] not in used_ticks, 'More than one pseudo per tick')
            used_ticks.add(row['parent_tick'])
            if row['modality'] == modality.lower():
                validate_cal_row(row, membership, modality)
                vector = arrays[modality.lower()][i]
                features.append(vector[None, :])
                labels.append(0)
                provenance.append(row)
    for row in provenance:
        validate_cal_row(row, membership, modality)
    return np.concatenate(features), np.array(labels, dtype=np.int64), provenance
