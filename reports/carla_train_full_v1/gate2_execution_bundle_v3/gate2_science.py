"""Execute exact bound scientific definitions in isolated globals, without core startup.

No AST bodies are edited. The local import resolver supplies the unchanged core
types to ensemble_to_uncertainty; it does not import cognix's broad __init__.
"""
import ast
import builtins
import dataclasses
import enum
import math
import sys
import types
import typing
from pathlib import Path
import numpy as np
from gate2_common import REPO, V8, require, hash_file, read_json, DIMS, MODALITIES

EXTRA_SOURCES = ('cognix/core/types.py', 'cognix/core/interfaces.py')


def definitions(path, names, ns):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    require({node.name for node in selected} == set(names), 'Missing scientific definition')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)


def load_science():
    science = read_json(V8 / 'scientific_source_hashes.json')
    for record in science['source_files']:
        require(hash_file(REPO / record['path']) == record['working_file_sha256'],
                'Scientific working source mismatch: ' + record['path'])
    bindings = read_json(Path(__file__).parent / 'execution_bindings.json')
    for path, sha in bindings['extra_source_hashes'].items():
        require(hash_file(REPO / path) == sha, 'Extra scientific source mismatch')
    name = '_gate2_frozen_science'
    module = types.ModuleType(name)
    sys.modules[name] = module  # dataclasses resolve their exact local namespace
    ns = module.__dict__
    ns.update({key: getattr(typing, key) for key in ('Any', 'Callable', 'Dict', 'List', 'Optional', 'Tuple')})
    ns.update(np=np, dataclass=dataclasses.dataclass, field=dataclasses.field,
              Enum=enum.Enum, Path=Path, cos=math.cos, radians=math.radians,
              degrees=math.degrees)
    definitions(REPO / 'cognix/core/types.py', ('PredictionResult', 'UncertaintyResult'), ns)
    # RealNormalityAgent uses the unchanged interface only as a base; load it
    # into its shared namespace to preserve abstract-method checks.
    import abc
    ns.update(ABC=abc.ABC, abstractmethod=abc.abstractmethod)
    definitions(REPO / 'cognix/core/interfaces.py', ('AgentInterface',), ns)
    original_import = builtins.__import__
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'cognix.core.types':
            return module
        require(not name.startswith(('cognix.', 'torch')), 'Unexpected scientific import')
        return original_import(name, globals, locals, fromlist, level)
    ns['__builtins__'] = dict(vars(builtins), __import__=local_import)
    definitions(REPO / 'cognix/adapters/carla/normality.py',
                ('NormalityModel', 'MahalanobisNormality', 'ScoreCalibrator',
                 'ensemble_jensen_diagnostic', 'score_separation', 'fixed_slope_profile',
                 'EnsemblePredictiveCalibrator', 'BootstrapNormalityEnsemble',
                 '_bernoulli_entropy', 'ensemble_to_uncertainty'), ns)
    definitions(REPO / 'cognix/adapters/carla/real_agents.py',
                ('RealAgentError', 'RealNormalityAgent', '_identity', 'RealCameraAgent',
                 'RealSegAgent', 'RealIMUAgent'), ns)
    definitions(REPO / 'cognix/adapters/carla/real_features.py',
                ('camera_embedding_features', 'segmentation_histogram_features',
                 'imu_window_features'), ns)
    definitions(REPO / 'cognix/adapters/carla/carlanomaly_loader.py', ('CarlAnomalySplit',), ns)
    ns['IMU_ACCEL_COLUMNS'] = ('acceleration_x', 'acceleration_y', 'acceleration_z')
    definitions(REPO / 'cognix/adapters/carla/pseudo_anomalies.py',
                ('PseudoAnomalyError', 'require_train_split', 'PseudoAnomalySample',
                 '_as_observation_dict', '_validated_float', '_camera_rect_darken',
                 'camera_brightness_shift', 'camera_occlusion', 'seg_region_corruption',
                 '_imu_accel_matrix', '_rebuild_imu_table', 'imu_spike', 'imu_bias_scale',
                 'generate_pseudo_anomaly'), ns)
    ns['_RECIPE_BY_ID'] = {rid: ns[rid] for rid in (
        'camera_brightness_shift', 'camera_occlusion', 'seg_region_corruption',
        'imu_spike', 'imu_bias_scale')}
    definitions(V8 / 'segmentation_oov.py', ('sanitize',), ns)
    ns['require'] = require
    return ns


def new_agent(science, modality):
    require(modality in MODALITIES, 'GNSS/foreign modality forbidden')
    return science['Real' + modality + 'Agent'](n_members=5, seed=42)


def check_features(x, modality):
    require(modality in DIMS and x.ndim == 2 and x.shape[1] == DIMS[modality],
            'Exact modality feature dimension mismatch')
    require(len(x) > 0 and np.isfinite(x).all(), 'Nonfinite/empty required features')


def member_arrays(agent):
    require(agent.n_members == 5 and agent.seed == 42 and
            len(agent._ensemble._members) == 5, 'Bootstrap contract mismatch')
    return {f'member_{i}_{key}': np.asarray(value) for i, member in
            enumerate(agent._ensemble._members) for key, value in (
                ('mean', member._mean), ('precision', member._prec),
                ('d2_scale', member._d2_scale), ('shrinkage', member.shrinkage))}


def restore_agent(science, modality, arrays):
    agent = new_agent(science, modality)
    ensemble = science['BootstrapNormalityEnsemble'](n_members=5,
                factory=science['MahalanobisNormality'], seed=42)
    for i in range(5):
        member = science['MahalanobisNormality']()
        member._mean = arrays[f'member_{i}_mean'].copy()
        member._prec = arrays[f'member_{i}_precision'].copy()
        member._d2_scale = float(arrays[f'member_{i}_d2_scale'])
        member.shrinkage = float(arrays[f'member_{i}_shrinkage'])
        require(member._mean.shape == (DIMS[modality],) and
                member._prec.shape == (DIMS[modality], DIMS[modality]) and
                member._d2_scale > 0 and member.shrinkage == .1, 'Invalid restored state domain')
        ensemble._members.append(member)
    require(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite restored state')
    agent._ensemble = ensemble
    return agent
