"""Inspect sealed real states without instantiation; restore ONLY in future data or synthetic context."""
import copy
import graph_common
from graph_common import *


def validate_member(values, modality):
    import numpy as np
    require(set(values)=={'mean','precision','d2_scale','shrinkage'},'Ambiguous fitted member keys')
    d=DIMS[modality]
    require(values['mean'].shape==(d,) and values['precision'].shape==(d,d) and
            values['d2_scale'].shape==values['shrinkage'].shape==(),'Wrong fitted state shape')
    require(all(v.dtype==np.float64 and np.isfinite(v).all() for v in values.values()) and
            float(values['d2_scale'])>0 and float(values['shrinkage'])==.1,'Invalid fitted state domain')


def validate_mapping(audit, parameters):
    import numpy as np
    require(audit['status']=='VALID' and audit['require_valid_succeeded'] is True and
            audit['calibrator_runtime_class']=='EnsemblePredictiveCalibrator (unchanged production agent)' and
            audit['optimization_info']['finite_optimum'] is True and audit['outputs_complete'] is True,
            'Accepted current mapping cannot be unambiguously restored')
    require(parameters.dtype==np.float64 and parameters.shape==(4,) and np.isfinite(parameters).all() and
            parameters[0]>0 and parameters[3]>0 and
            np.array_equal(parameters,np.asarray(audit['calibrator_parameters'],dtype=np.float64)),
            'JSON/NPZ mapping parameters disagree or domain fails')
    return dict(zip(('a','b','score_mean','score_std'),map(float,parameters)))


def inspect_current_states():
    """Reads fitted-state/mapping representation only; no feature scoring or objects restored."""
    import numpy as np
    result={}
    for modality in NODE_ORDER:
        fit=REPORT/'gate2_train_health_v3/units'/('fit-'+modality)
        unit=REPORT/'gate2_train_health_v3/units'/('audit-'+modality)
        members=[]
        for i in range(5):
            path=fit/f'member_{i}.npz'
            with np.load(safe_path(path),allow_pickle=False) as z:
                require(len(z.files)==len(set(z.files)),'Duplicate NPZ state keys')
                values={k:z[k] for k in z.files}
            validate_member(values,modality)
            members.append(dict(file=path.relative_to(REPO).as_posix(),sha256=hash_file(path),
                shapes={k:list(v.shape) for k,v in values.items()},dtypes={k:str(v.dtype) for k,v in values.items()}))
        audit=read_json(unit/'audit.json')
        with np.load(unit/'audit_arrays.npz',allow_pickle=False) as z:
            require(len(z.files)==len(set(z.files)),'Duplicate calibration NPZ keys')
            params=z['calibrator_parameters_a_b_score_mean_std']
        mapping=validate_mapping(audit,params)
        binding=read_json(fit/'fit_binding.json')
        require(binding['bootstrap_members']==5 and binding['bootstrap_seed']==42 and
                binding['total_rows']==76*2999 and binding['pseudo_rows']==binding['TEST_rows']==0,
                'Current clean-only fitted-state binding differs')
        result[modality]=dict(members=members,bootstrap_members=5,bootstrap_seed=42,
            audit_json_sha256=hash_file(unit/'audit.json'),audit_arrays_sha256=hash_file(unit/'audit_arrays.npz'),
            fit_binding_sha256=hash_file(fit/'fit_binding.json'),mapping=mapping,
            parameter_NPZ_key='calibrator_parameters_a_b_score_mean_std',
            parameter_order=['a','b','score_mean','score_std'],accepted_validity_status='VALID',
            mapping_equation='p_member = sigmoid(a * raw_member_score + b); q_normal = mean(p_member)',
            normalization='a and b are already raw-score parameters; score_mean/std restored for state identity only',
            UQ='total=H(mean(p)); aleatoric=mean(H(p)); epistemic=max(0,total-aleatoric)',
            restoration='object.__new__ of authenticated fitted member/ensemble/calibrator; copy exact sealed fields; no init/fit/calibrate',
            required_calibrator_fields=['a','b','_s_mean','_s_std','_fitted','_validity','_nll_initial','_nll_final','_n_iter','_converged'])
    return dict(status='PASS',modalities=result,real_objects_instantiated=False,real_features_scored=False,
                upstream_refit=False,recalibration=False,scientific_representation='float64')


class RestoredScorer:
    """Thin read-only facade over the exact production scoring/decomposition definitions."""
    def __init__(self,modality,members,audit,parameters,science,*,context):
        import numpy as np
        require(context in ('future-authorized-data','synthetic-fixture'),'Real restoration forbidden in preparation')
        require(graph_common.ACTIVE_PHASE == ('graph-data' if context=='future-authorized-data' else 'synthetic'),
                'Restoration requires a matching guarded process phase')
        require(modality in NODE_ORDER and len(members)==5,'Foreign modality/member count')
        for member in members: validate_member(member,modality)
        mapping=validate_mapping(audit,parameters)
        self.modality,self.science=modality,science
        cls=science['BootstrapNormalityEnsemble']
        ensemble=object.__new__(cls)
        ensemble.n_members,ensemble.seed,ensemble.factory,ensemble._members=5,42,science['MahalanobisNormality'],[]
        for values in members:
            member=object.__new__(science['MahalanobisNormality'])
            member._mean=values['mean'].copy();member._mean.flags.writeable=False
            member._prec=values['precision'].copy();member._prec.flags.writeable=False
            member._d2_scale=float(values['d2_scale']);member.shrinkage=float(values['shrinkage'])
            ensemble._members.append(member)
        calibrator=object.__new__(science['EnsemblePredictiveCalibrator'])
        calibrator.a,calibrator.b=mapping['a'],mapping['b']
        calibrator._s_mean,calibrator._s_std=mapping['score_mean'],mapping['score_std']
        calibrator._fitted=True;calibrator._validity=copy.deepcopy(audit['optimization_info'])
        for field in ('nll_initial','nll_final','n_iter','converged'):
            setattr(calibrator,'_'+field,audit['optimization_info'][field])
        calibrator.require_valid(diagnostic=False)
        self.ensemble,self.calibrator=ensemble,calibrator

    def node(self,features):
        import numpy as np
        vector=np.asarray(features)
        require(vector.dtype==np.float64 and vector.shape==(DIMS[self.modality],) and np.isfinite(vector).all(),
                'Invalid current scientific feature vector')
        scores=self.ensemble.predict_normality(vector)
        require(scores.shape==(5,) and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)),'Member score domain')
        probabilities=np.asarray([self.calibrator.prob_normal(float(s),diagnostic=False) for s in scores],dtype=np.float64)
        uq=self.science['ensemble_to_uncertainty'](probabilities)
        return np.array([float(probabilities.mean()),uq.epistemic,uq.aleatoric],dtype=np.float64)

    def fit(self,*args,**kwargs): require(False,'Upstream refit prohibited')
    def fit_calibrator(self,*args,**kwargs): require(False,'Recalibration prohibited')


def restore_current(science):
    require(graph_common.ACTIVE_PHASE=='graph-data','Current real restoration forbidden during preparation')
    import numpy as np
    inspect_current_states()  # fails before any object construction on ambiguity
    agents={}
    for modality in NODE_ORDER:
        fit=REPORT/'gate2_train_health_v3/units'/('fit-'+modality)
        unit=REPORT/'gate2_train_health_v3/units'/('audit-'+modality)
        members=[]
        for i in range(5):
            with np.load(fit/f'member_{i}.npz',allow_pickle=False) as z: members.append({k:z[k] for k in z.files})
        with np.load(unit/'audit_arrays.npz',allow_pickle=False) as z: parameters=z['calibrator_parameters_a_b_score_mean_std']
        agents[modality]=RestoredScorer(modality,members,read_json(unit/'audit.json'),parameters,science,
                                     context='future-authorized-data')
    return agents
