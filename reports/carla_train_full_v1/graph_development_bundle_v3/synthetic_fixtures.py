"""Explicit synthetic inputs; never loads a real clean block, CAL pseudo, or fitted state."""
import numpy as np
from graph_common import *
from current_upstream import authenticate_gate2
from current_restore import RestoredScorer
from graph_export import Admission,causal_window


def fixture_partition():
    fit=[f'fixture/fit-{i:03d}' for i in range(76)]
    return dict(FIT_NORMAL=fit,GRAPH_TRAIN=fit[:61],GRAPH_VAL=fit[61:],
                CAL_NORMAL_excluded=[f'fixture/cal-{i:03d}' for i in range(25)])


def science_fixture():
    return authenticate_gate2()['gate2_science'].load_science()


def fake_agents(science):
    agents={}
    for m in NODE_ORDER:
        d=DIMS[m]
        members=[dict(mean=np.full(d,.02*i,dtype=np.float64),precision=np.eye(d,dtype=np.float64),
                      d2_scale=np.array(4.,dtype=np.float64),shrinkage=np.array(.1,dtype=np.float64)) for i in range(5)]
        parameters=np.array([3.,-1.,.5,.2],dtype=np.float64)
        audit=dict(status='VALID',require_valid_succeeded=True,outputs_complete=True,
            calibrator_runtime_class='EnsemblePredictiveCalibrator (unchanged production agent)',
            calibrator_parameters=parameters.tolist(),optimization_info=dict(finite_optimum=True,status='VALID',
                nll_initial=.8,nll_final=.5,n_iter=3,converged=True))
        agents[m]=RestoredScorer(m,members,audit,parameters,science,context='synthetic-fixture')
    return agents


def raw_fixture():
    return dict(Camera=(np.arange(16*16*3).reshape(16,16,3)%190+30).astype(np.uint8),
        Seg=(np.arange(16*16).reshape(16,16)%29).astype(np.uint8),
        IMU=np.arange(40*3,dtype=np.float64).reshape(40,3)/100)


def compact_parent(science,raw,tick):
    return dict(Camera=science['camera_embedding_features'](raw['Camera']),
        Seg=science['segmentation_histogram_features'](science['sanitize'](raw['Seg'])[0]),
        IMU=science['imu_window_features'](causal_window(raw['IMU'],tick)))


def model_fixture():
    from graph_models import SyntheticGraphs
    rng=np.random.Generator(np.random.PCG64(17))
    nodes=rng.uniform(.05,.6,size=(46,3,3)).astype(np.float64)
    targets=np.asarray([1,0]*23,dtype=np.float32)
    ids=[f'fixture/fit-{i//2:03d}' for i in range(16)]+[f'fixture/val-{i//2:03d}' for i in range(30)]
    return SyntheticGraphs(nodes,targets,ids,[f'fixture:row-{i:03d}' for i in range(46)],
        ['GRAPH_TRAIN']*16+['GRAPH_VAL']*30)
