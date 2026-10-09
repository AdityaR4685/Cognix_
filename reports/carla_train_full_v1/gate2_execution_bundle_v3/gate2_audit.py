"""Frozen status decision order; descriptive ranges never become performance gates."""
import math
import numpy as np
from gate2_common import MODALITIES, SEMANTICS, require
from gate2_data import validate_cal_row

AUDIT_FIELDS = {'nll_initial', 'nll_final', 'n_iter', 'converged', 'optimizer_converged',
                'finite_optimum', 'suspected_separation', 'gradient_norm', 'minimum_curvature',
                'slope_raw', 'boundary_trend', 'lower_profile_direction', 'status', 'separation',
                'slope_profile', 'relative_slope_profile', 'validity_scope'}


def json_evidence(value):
    """Retain nonfinite evidence explicitly rather than emit invalid JSON or drop it."""
    if isinstance(value, (np.ndarray,)):
        return json_evidence(value.tolist())
    if isinstance(value, np.generic):
        return json_evidence(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else '+Infinity' if value > 0 else '-Infinity'}
    if isinstance(value, dict):
        return {str(k): json_evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_evidence(v) for v in value]
    return value


def summary(values):
    a = np.asarray(values, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    return {'count': len(a), 'finite_count': len(finite), 'nonfinite_count': int((~np.isfinite(a)).sum()),
            'min': float(finite.min()) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None,
            'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std()) if len(finite) else None,
            'quantiles': {str(q): float(np.quantile(finite, q)) if len(finite) else None
                          for q in (0, .1, .25, .5, .75, .9, 1)},
            'exact_zero_count': int((a == 0).sum()), 'exact_one_count': int((a == 1).sum())}


def constants(values):
    a = np.asarray(values)
    return {'exact_constant_output': bool(len(a) > 0 and np.all(a == a[0])),
            'exact_constant_columns': [bool(len(a) > 0 and np.all(a[:, j] == a[0, j]))
                                       for j in range(a.shape[1])] if a.ndim == 2 else None}


def audit_complete(info):
    if not isinstance(info, dict) or not AUDIT_FIELDS <= info.keys():
        return False
    try:
        geometry = info['separation']
        fields = {'normal_range', 'pseudo_range', 'overlap_interval', 'min_normal', 'max_pseudo',
                  'gap', 'best_threshold', 'threshold_rule', 'violations', 'classification'}
        return all(fields <= geometry[key].keys() for key in ('ensemble_mean', 'pooled_members')) and \
            len(info['slope_profile']) == 12 and len(info['relative_slope_profile']) == 3 and \
            all({'slope_raw', 'intercept_raw', 'nll'} <= p.keys() for p in
                info['slope_profile'] + info['relative_slope_profile']) and \
            all(type(info[k]) is bool for k in ('finite_optimum', 'suspected_separation',
                                               'lower_profile_direction', 'optimizer_converged', 'converged'))
    except (KeyError, TypeError):
        return False


def structural(info):
    if not isinstance(info, dict):
        return False
    pooled = info.get('separation', {}).get('pooled_members', {}).get('classification')
    return pooled in ('completely_separated', 'quasi_separated') or \
        info.get('suspected_separation') is True or info.get('boundary_trend') == 'decreasing' or \
        info.get('lower_profile_direction') is True


def classify(info, numeric_ok, provenance_ok, supported_classes, constant_output, require_valid_ok):
    if not numeric_ok:
        return 'INVALID_NUMERIC'
    if structural(info):
        return 'INVALID_STRUCTURAL'
    if not audit_complete(info) or info.get('finite_optimum') is not True or not provenance_ok or \
            not supported_classes or constant_output or not require_valid_ok:
        return 'INVALID_OTHER'
    return 'VALID'


def gate_decision(statuses):
    require(set(statuses) == set(MODALITIES) and all(v in
            ('VALID', 'INVALID_STRUCTURAL', 'INVALID_NUMERIC', 'INVALID_OTHER')
            for v in statuses.values()), 'Missing/foreign Gate-2 modality status')
    return 'TRAIN_ONLY_SCIENTIFIC_HEALTH_' + ('PASS' if all(v == 'VALID' for v in statuses.values()) else 'FAIL')


def numeric_predicates(features, scores, probabilities, uq, parameters):
    groups = {'features': features, 'member_scores': scores, 'member_probabilities': probabilities,
              'UQ': uq, 'parameters': parameters}
    finite = {key: bool(np.isfinite(np.asarray(value, dtype=float)).all()) for key, value in groups.items()}
    domain = {'member_scores_in_0_1': bool(np.all((scores >= 0) & (scores <= 1))),
              'member_probabilities_in_0_1': bool(np.all((probabilities >= 0) & (probabilities <= 1))),
              'UQ_nonnegative_nats': bool(np.all(uq >= 0)),
              'UQ_entropy_upper_bound': bool(np.all(uq <= math.log(2) + 1e-12)),
              'aleatoric_and_epistemic_bounded_by_total': bool(np.all(uq[:, 1:] <= uq[:, :1] + 1e-12)),
              'entropy_decomposition_identity': bool(np.allclose(uq[:, 0], uq[:, 1] + uq[:, 2], rtol=0, atol=1e-12)),
              'positive_slope': bool(parameters[0] > 0),
              'positive_score_standard_deviation': bool(parameters[3] > 0)}
    return {'finite': finite, 'domain': domain, 'pass': all(finite.values()) and all(domain.values())}


def audit_modality(science, agent, modality, features, labels, provenance, fit_binding, membership):
    scores = np.array([agent.predict_normality(row) for row in features])
    info, params, exception = None, None, None
    probabilities = np.full(scores.shape, np.nan)
    uq = np.full((len(features), 3), np.nan)
    probability_available = np.zeros(len(features), dtype=bool)
    uq_available = np.zeros(len(features), dtype=bool)
    valid_ok = False
    supported = set(np.unique(labels).tolist()) == {0, 1}
    if np.isfinite(features).all() and np.isfinite(scores).all() and supported:
        try:
            agent.fit_calibrator(scores, labels)
            calibrator = agent._calibrator
            info = calibrator.optimization_info
            params = np.array([calibrator.a, calibrator.b, calibrator._s_mean, calibrator._s_std])
            for i, member in enumerate(scores):
                # Invalid fitted channels are evaluated exclusively for audit evidence.
                probabilities[i] = [calibrator.prob_normal(float(s), diagnostic=True) for s in member]
                probability_available[i] = True
                u = science['ensemble_to_uncertainty'](probabilities[i])
                uq[i] = [u.total, u.aleatoric, u.epistemic]
                uq_available[i] = True
            try:
                calibrator.require_valid(diagnostic=False)
                valid_ok = True
            except RuntimeError:
                pass
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            exception = repr(exc)
    q = probabilities.mean(axis=1)
    if params is not None:
        numeric = numeric_predicates(features, scores, probabilities[probability_available], uq[uq_available], params)
    else:
        # Missing outputs are incomplete evidence, not fabricated numeric NaNs.
        numeric = {'finite': {'features': bool(np.isfinite(features).all()),
                              'member_scores': bool(np.isfinite(scores).all())},
                   'domain': {'member_scores_in_0_1': bool(np.all((scores >= 0) & (scores <= 1)))}}
        numeric['pass'] = all(numeric['finite'].values()) and all(numeric['domain'].values())
        if exception and any(term in exception.lower() for term in ('non-finite', 'nonfinite', 'overflow')):
            numeric['pass'] = False
    numeric['upstream_state_finite'] = all(np.isfinite(v).all() for v in __import__('gate2_science').member_arrays(agent).values())
    numeric['pass'] &= numeric['upstream_state_finite']
    def finite_audit(value):
        if isinstance(value, (float, int)):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_audit(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_audit(v) for v in value)
        return True
    numeric['optimizer_audit_finite'] = finite_audit(info)
    numeric['pass'] &= numeric['optimizer_audit_finite']
    outputs_complete = bool(probability_available.all() and uq_available.all() and params is not None)
    provenance_ok = len(provenance) == len(features) == len(labels) and fit_binding['pseudo_rows'] == 0 and \
        fit_binding['TEST_rows'] == 0 and all(row['source_split'] == 'train' and
        row['partition'] == 'CAL_NORMAL' and row['upstream_fit'] is False and row['graph_role'] is None
        for row in provenance)
    try:
        for row, label in zip(provenance, labels):
            validate_cal_row(row, membership, modality)
            provenance_ok &= row['target_normal'] == int(label)
    except (RuntimeError, KeyError, TypeError, ValueError):
        provenance_ok = False
    provenance_ok &= outputs_complete
    constant = constants(q)['exact_constant_output'] if outputs_complete else False
    status = classify(info, numeric['pass'], provenance_ok, supported, constant, valid_ok)
    saturation = lambda a: {'exact_zero_count': int((a == 0).sum()), 'exact_one_count': int((a == 1).sum()),
        'exact_saturation_fraction': float(((a == 0) | (a == 1)).mean()) if a.size else None,
        'within_1e_12_boundary_count': int(((a <= 1e-12) | (a >= 1 - 1e-12)).sum()),
        'within_1e_12_boundary_fraction': float(((a <= 1e-12) | (a >= 1 - 1e-12)).mean()) if a.size else None,
        'near_boundary_threshold_is_diagnostic_only': True}
    report = {'modality': modality, 'status': status, 'prediction_semantics': SEMANTICS,
        'calibrator_contract_label': 'ScoreCalibrator in frozen gate text',
        'calibrator_runtime_class': 'EnsemblePredictiveCalibrator (unchanged production agent)',
        'optimization_info': info, 'calibrator_parameters': params, 'calibration_exception': exception,
        'require_valid_succeeded': valid_ok, 'numeric_predicates': numeric,
        'outputs_complete': outputs_complete,
        'availability': {'probability_rows': int(probability_available.sum()), 'UQ_rows': int(uq_available.sum()),
                         'parameters_available': params is not None,
                         'unavailable_array_entries_use_NaN_placeholders': True,
                         'numeric_checks_and_summaries_exclude_unavailable_placeholders': True},
        'provenance_complete': provenance_ok, 'supported_CAL_classes': supported,
        'sample_counts': {'FIT_clean': fit_binding['total_rows'], 'CAL_combined': len(labels),
                          'CAL_clean': int((labels == 1).sum()), 'CAL_pseudo': int((labels == 0).sum())},
        'score_range_and_finite_counts': summary(scores), 'probability_range_and_finite_counts': summary(probabilities[probability_available]),
        'ensemble_probability': summary(q[probability_available]), 'probability_saturation': saturation(probabilities[probability_available]),
        'ensemble_saturation': saturation(q[probability_available]), 'feature_finite_counts': summary(features),
        'member_score_disagreement': {'std': summary(scores.std(axis=1)), 'range': summary(np.ptp(scores, axis=1))},
        'member_probability_disagreement': {'std': summary(probabilities[probability_available].std(axis=1)), 'range': summary(np.ptp(probabilities[probability_available], axis=1))},
        'uncertainty_units': 'natural-log nats', 'epistemic_amplification': False,
        'uncertainty': {key: summary(uq[uq_available, j]) for j, key in enumerate(('total', 'aleatoric', 'epistemic'))},
        'exact_constant_checks': {'features': constants(features), 'member_scores': constants(scores),
                                  'mean_score': constants(scores.mean(axis=1)),
                                  'member_probabilities': constants(probabilities), 'predictive_output': constants(q)},
        'class_distributions': {name: {'scores': summary(scores[labels == label]),
                                     'probabilities': summary(probabilities[(labels == label) & probability_available]),
                                     'prediction': summary(q[(labels == label) & probability_available]),
                                     'UQ': {key: summary(uq[(labels == label) & uq_available, j]) for j, key in
                                            enumerate(('total', 'aleatoric', 'epistemic'))}}
                                for name, label in (('clean', 1), ('pseudo', 0))},
        'recipe_roles': {rid: sum(row.get('recipe_id') == rid for row in provenance) for rid in
                         sorted({r.get('recipe_id') for r in provenance if r.get('recipe_id')})},
        'FIT_binding': fit_binding, 'CAL_parent_scenarios': sorted({r['parent_scenario'] for r in provenance}),
        'scenario_isolation': provenance_ok, 'diagnostic_only_if_invalid': status != 'VALID',
        'pseudo_discrimination_interpretation': 'TRAIN constructed-task development diagnostic only',
        'decision_noncriteria': ['no AUROC minimum', 'no ECE improvement minimum', 'no epistemic minimum',
                                'no probability range minimum', 'no pseudo-discrimination quality requirement']}
    arrays = {'features': features, 'target_normal': labels, 'member_scores': scores,
              'member_probabilities': probabilities, 'mean_probability': q,
              'probability_row_available': probability_available, 'UQ_row_available': uq_available,
              'total_aleatoric_epistemic_nats': uq}
    if params is not None:
        arrays['calibrator_parameters_a_b_score_mean_std'] = params
    return json_evidence(report), arrays
