"""Mathematical calibration-validity tests; no benchmark metrics or raw data."""
import numpy as np
import pytest
from cognix.adapters.carla.normality import EnsemblePredictiveCalibrator, score_separation
from cognix.adapters.carla.real_agents import RealCameraAgent, RealAgentError

def finite_fixture(kind='seg'):
    # Identical members reduce exactly to scalar logistic likelihood with a
    # finite positive MLE. Both labels occur at every level.
    s=np.repeat([.1,.3,.5,.7,.9],10)
    counts=[1,3,5,7,9] if kind=='seg' else [2,4,5,6,8]
    y=np.concatenate([np.r_[np.ones(n),np.zeros(10-n)] for n in counts])
    return np.repeat(s[:,None],5,axis=1),y

@pytest.mark.parametrize('kind',['seg','imu'])
def test_finite_seg_imu_mathematical_fixtures(kind):
    M,y=finite_fixture(kind); c=EnsemblePredictiveCalibrator().fit(M,y)
    assert c.optimization_info['finite_optimum']
    assert not c.optimization_info['suspected_separation']
    assert c.optimization_info['gradient_norm']<1e-8
    assert c.prob_normal(.8)>c.prob_normal(.2)
    assert np.isfinite([c.prob_normal(s) for s in [-1e6,0,1,1e6]]).all()

@pytest.mark.parametrize('kind',['camera','gnss'])
def test_separated_fixtures_refuse_scientific_probabilities(kind):
    M=np.repeat(np.array([.1,.2,.8,.9])[:,None],5,axis=1)
    if kind=='gnss': M*=1e-6
    y=np.array([0,0,1,1]); c=EnsemblePredictiveCalibrator().fit(M,y)
    assert c.optimization_info['suspected_separation']
    assert not c.optimization_info['finite_optimum']
    assert c.optimization_info['separation']['pooled_members']['classification']=='completely_separated'
    with pytest.raises(RuntimeError,match='invalid for scientific use'): c.prob_normal(.5)
    with pytest.raises(RuntimeError,match='invalid for scientific use'): c.ensemble_probability(M[0])
    assert 0<=c.prob_normal(.5,diagnostic=True)<=1

def test_quasi_separation_exact_shared_boundary():
    M=np.array([.1,.5,.5,.9]); y=np.array([0,0,1,1])
    c=EnsemblePredictiveCalibrator().fit(M,y)
    assert c.optimization_info['separation']['pooled_members']['classification']=='quasi_separated'
    assert c.optimization_info['suspected_separation']
    assert not c.optimization_info['finite_optimum']

def test_mean_separation_is_not_member_separation_proof():
    M=np.array([[0,.6],[.4,1.]])
    g=score_separation(M,np.array([0,1]))
    assert g['ensemble_mean']['classification']=='completely_separated'
    assert g['pooled_members']['classification']=='overlapping'

def test_permutation_repeated_fit_and_duplicate_weight_invariance():
    M,y=finite_fixture(); c=EnsemblePredictiveCalibrator().fit(M,y)
    for other in [M.copy(),M[:,::-1],np.tile(M,(1,3))]:
        d=EnsemblePredictiveCalibrator().fit(other,y)
        assert d.optimization_info['finite_optimum']
        assert d.a==pytest.approx(c.a,rel=1e-7)
        assert d.b==pytest.approx(c.b,abs=1e-7)
    d=EnsemblePredictiveCalibrator().fit(M,y)
    assert d.optimization_info==c.optimization_info

@pytest.mark.parametrize('labels',[np.zeros(4),np.ones(4)])
def test_validity_requires_both_classes(labels):
    with pytest.raises(ValueError,match='both classes'):
        EnsemblePredictiveCalibrator().fit(np.arange(4),labels)

def test_invalid_real_agent_blocks_prediction_and_uq_with_auditable_diagnostics():
    agent=RealCameraAgent().fit(np.random.default_rng(0).normal(size=(30,3)))
    agent.fit_calibrator(np.array([.1,.2,.8,.9]),np.array([0,0,1,1]))
    for method in [agent.predict,agent.estimate_uncertainty]:
        with pytest.raises(RealAgentError,match='invalid for scientific use'): method(np.zeros(3))
    p=agent.predict(np.zeros(3),diagnostic=True)
    assert p.metadata['diagnostic_only'] and not p.metadata['calibration_valid']
    assert p.metadata['prediction_semantics']=='P(target=normal) under TRAIN-derived clean-vs-pseudo calibration'
    assert agent.metadata()['calibration_audit']['suspected_separation']

def test_failed_refit_clears_valid_mapping():
    M,y=finite_fixture(); c=EnsemblePredictiveCalibrator().fit(M,y)
    with pytest.raises(ValueError): c.fit(M,np.ones(len(y)))
    assert not c.is_fitted
    with pytest.raises(RuntimeError,match='not fitted'): c.prob_normal(.5)

def test_constant_member_rows_have_no_identifiable_slope():
    c=EnsemblePredictiveCalibrator().fit(np.ones((6,3)),np.array([0,0,0,1,1,1]))
    assert not c.optimization_info['finite_optimum']

def test_valid_real_agent_default_prediction_and_uq():
    M,y=finite_fixture(); agent=RealCameraAgent().fit(np.random.default_rng(0).normal(size=(30,3)))
    agent.fit_calibrator(M,y)
    p=agent.predict(np.zeros(3)); u=agent.estimate_uncertainty(np.zeros(3))
    assert p.metadata['calibration_valid'] and not p.metadata['diagnostic_only']
    assert u.prediction==pytest.approx(p.value)

def test_legacy_cache_target_migrates_only_in_memory(tmp_path):
    import json
    from cognix.adapters.carla.cache_builder import load_cache
    path=tmp_path/'carla_train_cache.npz'
    legacy='cal_is_'+'safe'
    np.savez(path,**{legacy:np.array([0,1])})
    (tmp_path/'carla_train_cache.json').write_text(json.dumps({'cache_schema_version':1,'cache_content_sha256':'original'}))
    original=path.read_bytes()
    arrays,manifest=load_cache(tmp_path)
    assert legacy not in arrays
    assert np.array_equal(arrays['cal_target_normal'],[0,1])
    assert manifest['cache_content_sha256']=='original'
    assert path.read_bytes()==original

def test_finite_profile_has_a_turn_instead_of_monotonic_boundary_descent():
    M,y=finite_fixture(); c=EnsemblePredictiveCalibrator().fit(M,y)
    p=c.optimization_info['slope_profile']
    assert p[-1]['nll']>p[3]['nll']
    assert c.optimization_info['boundary_trend']=='not_decreasing'

def test_inverted_target_does_not_validate_zero_slope_boundary():
    M,y=finite_fixture(); c=EnsemblePredictiveCalibrator().fit(M,1-y)
    assert not c.optimization_info['finite_optimum']

def test_agent_refit_normal_model_invalidates_old_calibrator():
    M,y=finite_fixture(); x=np.random.default_rng(0).normal(size=(30,3))
    agent=RealCameraAgent().fit(x).fit_calibrator(M,y)
    agent.fit(x)
    with pytest.raises(RealAgentError,match='not ready'): agent.predict(x[0])
