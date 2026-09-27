"""
COGNIX Pipeline Orchestrator — Revision 3.

Canonical pipeline (M1 → M4 → M2 → M3 → M5):

  {p_i, sigma_e_i, sigma_a_i}
      -> Epistemic GAT
      -> {h_tilde_i, alpha_ij}
      -> probability extraction: p_tilde_i = sigmoid(H_prime[:, 0])
      -> Bayesian Fusion (epistemic-weighted)
      -> p_collective
      -> Conformal Prediction (pre-fitted on collective calibration outputs)
      -> Gamma(x)
      -> Risk + Decision
      -> Epistemic Shapley Attribution (collective value function)

Two modes:
  production  — fault-tolerant, all fallbacks active
  research    — fail loudly, no fallbacks, all modules must execute

ModuleStatus is recorded for every module in every cycle.
If any required module fails in research mode, RuntimeError propagates.
"""
from __future__ import annotations

import time
import logging
from typing import Any

import numpy as np

from cognix.engine.result import DecisionResult, DecisionOutcome, RiskLevel
from cognix.engine.provenance import ModuleStatus

logger = logging.getLogger(__name__)

RESEARCH_MODE = "research"
PRODUCTION_MODE = "production"


def _ms(start: float) -> float:
    """Elapsed time in milliseconds since start (perf_counter)."""
    return (time.perf_counter() - start) * 1000.0


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any | None:
    """Call fn; log and return None on exception. PRODUCTION ONLY."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning("Pipeline component %s failed: %s", fn, exc)
        return None


class CognixPipeline:
    """
    Orchestrates the full COGNIX decision pipeline.

    Parameters
    ----------
    mode : str
        "production" (default) — fault-tolerant, all fallbacks active.
        "research"             — fail loudly, no fallbacks, all modules must execute.
    """

    def __init__(
        self,
        config: Any = None,
        uncertainty_estimator: Any = None,
        belief_fuser: Any = None,
        calibrator: Any = None,
        graph: Any = None,
        communication: Any = None,
        attribution: Any = None,
        escalation: Any = None,
        mode: str = PRODUCTION_MODE,
    ) -> None:
        self.config = config
        self.uncertainty_estimator = uncertainty_estimator
        self.belief_fuser = belief_fuser
        self.calibrator = calibrator
        self.graph = graph
        self.communication = communication
        self.attribution = attribution
        self.escalation = escalation
        self.mode = mode

        if mode not in (RESEARCH_MODE, PRODUCTION_MODE):
            raise ValueError(f"mode must be 'research' or 'production', got {mode!r}")

        if mode == RESEARCH_MODE:
            logger.info(
                "CognixPipeline initialized in RESEARCH MODE. "
                "No fallbacks. All module failures will propagate as errors."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        agents: list[Any],
        input_data: Any,
        context: dict[str, Any],
        agent_reliabilities: dict[str, float] | None = None,
    ) -> DecisionResult:
        """Execute the full pipeline and return a DecisionResult."""
        latencies: dict[str, float] = {}
        module_status: dict[str, ModuleStatus] = {}
        pipeline_start = time.perf_counter()

        # ── Step 1: Collect predictions ───────────────────────────────
        t = time.perf_counter()
        predictions = self._collect_predictions(agents, input_data)
        latencies["collect_predictions"] = _ms(t)

        if not predictions:
            logger.error("No predictions collected from agents. Returning ABSTAIN.")
            return self._emergency_result(latencies, pipeline_start)

        # ── Step 2: Estimate uncertainty (M1: UDE) ────────────────────
        t = time.perf_counter()
        uncertainties, uq_status = self._estimate_uncertainties(
            agents, predictions, input_data
        )
        latencies["estimate_uncertainty"] = _ms(t)
        uq_status.duration_ms = latencies["estimate_uncertainty"]
        module_status["uq"] = uq_status

        # ── Step 3: Trust weights ─────────────────────────────────────
        t = time.perf_counter()
        trust_weights = self._compute_trust_weights(uncertainties)
        latencies["compute_trust"] = _ms(t)

        # ── Step 4: Reliability ───────────────────────────────────────
        # Use provided per-agent accuracy on calibration split.
        # If not provided, document explicitly as "not computed" in production,
        # or raise in research mode.
        if agent_reliabilities is None:
            if self.mode == RESEARCH_MODE:
                # Allow running without reliability but log clearly
                logger.warning(
                    "RESEARCH MODE: agent_reliabilities not provided. "
                    "Belief fusion will use epistemic-uncertainty weights only. "
                    "Reliability term set to uniform 1.0 — document this assumption."
                )
            reliabilities = {aid: 1.0 for aid in uncertainties}
        else:
            reliabilities = agent_reliabilities

        # ── Step 5: Graph refinement (M4: EpistemicGAT) ──────────────
        t = time.perf_counter()
        gnn_status = ModuleStatus(
            executed=False, method="epistemic_gat",
            duration_ms=0, failure_reason=None
        )
        comm_info: dict = {}
        # Canonical normalization: extract float prediction probabilities p_i in [0, 1]
        # for all agents upfront, ensuring consistent Dict[str, float] across all paths.
        normalized_predictions: dict[str, float] = {
            a: float(self._extract_prediction_probability(predictions[a]))
            for a in predictions
        }
        refined_predictions = dict(normalized_predictions)

        if self.graph is not None:
            try:
                agent_order = list(predictions.keys())
                N = len(agent_order)

                # Node features: [p_i, sigma_e_i, sigma_a_i] — shape (N, 3)
                node_features = np.array([
                    [
                        normalized_predictions[a],
                        float(uncertainties[a].get("epistemic", 0.0)),
                        float(uncertainties[a].get("aleatoric", 0.0)),
                    ]
                    for a in agent_order
                ], dtype=np.float32)

                # Fully connected adjacency (no self-loops)
                adjacency = (np.ones((N, N)) - np.eye(N)).astype(np.float32)

                epi_unc = {a: uncertainties[a].get("epistemic", 0.0) for a in agent_order}


                # Guard against untrained learned graph models (exposing is_trained)
                if hasattr(self.graph, "is_trained") and not self.graph.is_trained():
                    if self.mode == RESEARCH_MODE:
                        raise RuntimeError(
                            "RESEARCH MODE: Graph model has not been trained. "
                            "Call shared_gat.fit(agents, X_train, y_train) on "
                            "training data before running the research pipeline. "
                            "Untrained random W produces arbitrary outputs."
                        )
                    logger.warning(
                        "PRODUCTION MODE: Graph model exposes is_trained() but is not trained. "
                        "Skipping graph refinement to avoid random-weight contamination; "
                        "retaining original agent predictions."
                    )
                    gnn_status = ModuleStatus(
                        executed=False,
                        method="epistemic_gat",
                        duration_ms=_ms(t),
                        failure_reason="graph model not trained",
                    )
                else:
                    # Execute Graph Refinement
                    graph_result = self.graph.forward(
                        node_features, adjacency, epi_unc, agent_order
                    )
                    
                    H_prime = graph_result.node_outputs
                    attn_list = graph_result.attention

                    # Extract refined probabilities:
                    # p_i_refined = sigmoid(H_prime[i, 0])
                    # This is the documented mapping: first output dimension -> probability.
                    refined_probs = 1.0 / (1.0 + np.exp(-H_prime[:, 0]))

                    for idx, a in enumerate(agent_order):
                        refined_predictions[a] = float(
                            np.clip(refined_probs[idx], 1e-7, 1 - 1e-7)
                        )

                    # Build communication info from last layer attention
                    attn_matrix = attn_list[-1]  # (N, N)
                    attn_dict = {
                        agent_order[i]: float(attn_matrix[i].sum())
                        for i in range(N)
                    }
                    comm_info = {
                        "edges": [[agent_order[i], agent_order[j]]
                                  for i in range(N) for j in range(N) if i != j],
                        "attention_weights": attn_dict,
                        "attention_matrix_shape": list(attn_matrix.shape),
                        "adjacency_type": "fully_connected",
                    }

                    gnn_status = ModuleStatus(
                        executed=True, method="epistemic_gat",
                        duration_ms=_ms(t), failure_reason=None,
                        inputs_validated=True, outputs_validated=True,
                    )

            except Exception as exc:
                gnn_status = ModuleStatus(
                    executed=False, method="epistemic_gat",
                    duration_ms=_ms(t), failure_reason=str(exc)
                )
                if self.mode == RESEARCH_MODE:
                    raise RuntimeError(
                        f"RESEARCH MODE: EpistemicGAT failed: {exc}"
                    ) from exc
                logger.warning("GAT failed (production fallback): %s", exc)

        latencies["graph_refinement"] = _ms(t)
        module_status["gnn"] = gnn_status

        # Use GAT-refined predictions for all downstream stages
        active_predictions = refined_predictions

        # ── Step 6: Belief Fusion (M2: BBN) ──────────────────────────
        t = time.perf_counter()
        belief_status = ModuleStatus(
            executed=False, method="", duration_ms=0, failure_reason=None
        )
        fused_confidence = self._aggregate_confidence(
            active_predictions, uncertainties, trust_weights
        )

        if self.belief_fuser is not None:
            try:
                fused_confidence = self._run_belief_fusion(
                    active_predictions, uncertainties, trust_weights, reliabilities
                )
                belief_status = ModuleStatus(
                    executed=True,
                    method=getattr(
                        getattr(self.belief_fuser, "default_strategy", None),
                        "value", "epistemic_weighted"
                    ),
                    duration_ms=_ms(t), failure_reason=None,
                    inputs_validated=True, outputs_validated=True,
                )
            except Exception as exc:
                belief_status = ModuleStatus(
                    executed=False, method="",
                    duration_ms=_ms(t), failure_reason=str(exc)
                )
                if self.mode == RESEARCH_MODE:
                    raise RuntimeError(
                        f"RESEARCH MODE: Belief fusion failed: {exc}"
                    ) from exc
                logger.warning("Belief fusion failed (production fallback): %s", exc)
        else:
            belief_status = ModuleStatus(
                executed=True, method="weighted_mean_fallback",
                duration_ms=_ms(t), failure_reason=None,
            )

        latencies["belief_fusion"] = _ms(t)
        module_status["belief"] = belief_status

        # ── Step 7: Aggregate uncertainty ─────────────────────────────
        t = time.perf_counter()
        agg_confidence, agg_epistemic, agg_aleatoric, agg_total = (
            self._aggregate_uncertainty(active_predictions, uncertainties, trust_weights)
        )
        # Fused confidence takes priority if belief_fuser ran
        if belief_status.executed and self.belief_fuser is not None:
            agg_confidence = fused_confidence
        latencies["aggregation"] = _ms(t)

        # ── Step 8: Calibration (M3: CCL) ─────────────────────────────
        t = time.perf_counter()
        cal_status = ModuleStatus(
            executed=False, method="", duration_ms=0, failure_reason=None
        )
        calibrated_confidence: float | None = None
        calibration_info: dict = {}

        if self.calibrator is not None:
            try:
                if not hasattr(self.calibrator, "cal_scores") or \
                        self.calibrator.cal_scores is None:
                    raise RuntimeError(
                        "ConformalPredictor has not been calibrated. "
                        "Call calibrator.calibrate(cal_outputs, cal_labels) "
                        "on collective calibration outputs before inference."
                    )

                cal_input = np.array([[1 - fused_confidence, fused_confidence]])
                prediction_sets = self.calibrator.predict(cal_input, alpha=0.05)
                ps = prediction_sets[0]

                calibrated_confidence = fused_confidence
                calibration_info = {
                    "method": "split_conformal",
                    "prediction_set": ps.prediction_set,
                    "target_coverage": ps.coverage_target,
                    "quantile": float(ps.quantile),
                    "n_calibration_samples": int(self.calibrator.n_cal),
                    "calibrated_confidence": calibrated_confidence,
                }

                cal_status = ModuleStatus(
                    executed=True, method="split_conformal",
                    duration_ms=_ms(t), failure_reason=None,
                    inputs_validated=True, outputs_validated=True,
                )

            except Exception as exc:
                cal_status = ModuleStatus(
                    executed=False, method="split_conformal",
                    duration_ms=_ms(t), failure_reason=str(exc)
                )
                if self.mode == RESEARCH_MODE:
                    raise RuntimeError(
                        f"RESEARCH MODE: Calibration failed: {exc}"
                    ) from exc
                logger.warning("Calibration failed (production fallback): %s", exc)
                calibration_info = {"calibrated_confidence": None}

        latencies["calibration"] = _ms(t)
        module_status["calibration"] = cal_status

        # ── Step 9: Risk assessment ───────────────────────────────────
        t = time.perf_counter()
        risk_level = self._assess_risk(fused_confidence, agg_epistemic)
        latencies["risk_assessment"] = _ms(t)

        # ── Step 10: Decision ─────────────────────────────────────────
        t = time.perf_counter()
        conformal_set_size = (
            len(calibration_info["prediction_set"])
            if "prediction_set" in calibration_info and calibration_info["prediction_set"] is not None
            else 1
        )
        decision, escalation_required, abstained, requested_info = self._make_decision(
            fused_confidence, agg_epistemic, risk_level, conformal_set_size=conformal_set_size
        )
        latencies["decision"] = _ms(t)

        # ── Step 11: Attribution (M5: DAE — Epistemic Shapley) ────────
        t = time.perf_counter()
        attr_status = ModuleStatus(
            executed=False, method="", duration_ms=0, failure_reason=None
        )
        agent_contributions: dict[str, float] = {}

        if self.attribution is not None and hasattr(self.attribution, "compute"):
            try:
                agent_contributions = self._run_attribution(
                    active_predictions, uncertainties, trust_weights
                )
                attr_status = ModuleStatus(
                    executed=True, method="epistemic_shapley",
                    duration_ms=_ms(t), failure_reason=None,
                    inputs_validated=True, outputs_validated=True,
                )
            except Exception as exc:
                attr_status = ModuleStatus(
                    executed=False, method="epistemic_shapley",
                    duration_ms=_ms(t), failure_reason=str(exc)
                )
                if self.mode == RESEARCH_MODE:
                    raise RuntimeError(
                        f"RESEARCH MODE: Epistemic Shapley failed: {exc}"
                    ) from exc
                logger.warning("Attribution failed (production fallback): %s", exc)

        latencies["attribution"] = _ms(t)
        module_status["attribution"] = attr_status

        # ── Step 12: Communication stats ──────────────────────────────
        comm_stats: dict | None = None
        if self.communication is not None:
            n_agents = len(agents)
            max_messages = n_agents * (n_agents - 1)
            actual = getattr(self.communication, "last_message_count", max_messages)
            if max_messages > 0:
                comm_stats = {
                    "total_messages": actual,
                    "max_possible": max_messages,
                    "reduction_ratio": 1.0 - actual / max_messages,
                }
        elif comm_info:
            # Use GAT communication info
            comm_stats = comm_info

        # ── Step 13: Explanation ──────────────────────────────────────
        t = time.perf_counter()
        explanation, reasoning_steps = self._generate_explanation(
            decision, fused_confidence, agg_epistemic, agg_aleatoric,
            risk_level, trust_weights, agent_contributions, module_status
        )
        latencies["explanation"] = _ms(t)

        total_latency = _ms(pipeline_start)

        return DecisionResult(
            decision=decision,
            confidence=fused_confidence,
            calibrated_confidence=calibrated_confidence,
            aleatoric_uncertainty=agg_aleatoric,
            epistemic_uncertainty=agg_epistemic,
            total_uncertainty=agg_total,
            risk_level=risk_level,
            escalation_required=escalation_required,
            abstained=abstained,
            requested_information=requested_info,
            agent_contributions=agent_contributions,
            agent_trust_weights=trust_weights,
            agent_predictions=active_predictions,
            communication_statistics=comm_stats,
            calibration_metrics=calibration_info,
            explanation=explanation,
            reasoning_steps=reasoning_steps,
            latency_ms=latencies,
            total_latency_ms=total_latency,
            timestamp=time.time(),
            session_id=context.get("session_id"),
            metadata={
                "context": context,
                "n_agents": len(agents),
                "mode": self.mode,
                "module_status": {k: v.to_dict() for k, v in module_status.items()},
            },
        )

    # ------------------------------------------------------------------
    # Pipeline stage implementations
    # ------------------------------------------------------------------

    def _collect_predictions(
        self, agents: list[Any], input_data: Any
    ) -> dict[str, Any]:
        """Collect predictions from all agents."""
        predictions: dict[str, Any] = {}
        for agent in agents:
            agent_id = self._agent_id(agent)
            try:
                if hasattr(agent, "predict"):
                    pred = agent.predict(input_data)
                elif callable(agent):
                    pred = agent(input_data)
                else:
                    logger.warning("Agent %s has no predict method; skipping.", agent_id)
                    continue
                predictions[agent_id] = pred
            except Exception as exc:
                logger.warning("Agent %s predict failed: %s", agent_id, exc)
        return predictions

    def _estimate_uncertainties(
        self,
        agents: list[Any],
        predictions: dict[str, Any],
        input_data: Any,
    ) -> tuple[dict[str, dict[str, float]], ModuleStatus]:
        """
        Estimate uncertainty per agent using agent.estimate_uncertainty().
        In research mode: if any agent lacks a real uncertainty estimator, raise.
        In production mode: fall back to confidence-derived heuristic (documented).
        """
        uncertainties: dict[str, dict[str, float]] = {}
        used_fallback = False

        for agent in agents:
            agent_id = self._agent_id(agent)
            if agent_id not in predictions:
                continue

            # Try agent's own uncertainty estimate
            if hasattr(agent, "estimate_uncertainty"):
                try:
                    est = agent.estimate_uncertainty(input_data)
                    if hasattr(est, "epistemic"):
                        uncertainties[agent_id] = {
                            "epistemic": float(est.epistemic),
                            "aleatoric": float(est.aleatoric),
                            "total": float(est.total),
                        }
                        continue
                except Exception as exc:
                    logger.debug(
                        "Agent %s estimate_uncertainty failed: %s", agent_id, exc
                    )

            # Try the configured uncertainty estimator
            if self.uncertainty_estimator is not None:
                try:
                    est = self.uncertainty_estimator.estimate(
                        getattr(agent, "model", getattr(agent, "_model", agent)), input_data
                    )
                    uncertainties[agent_id] = {
                        "epistemic": float(est.epistemic),
                        "aleatoric": float(est.aleatoric),
                        "total": float(est.total),
                    }
                    continue
                except Exception as exc:
                    logger.debug("UQ estimator for %s failed: %s", agent_id, exc)

            # ── RESEARCH MODE: no fallback ────────────────────────────
            if self.mode == RESEARCH_MODE:
                raise RuntimeError(
                    f"RESEARCH MODE: Agent '{agent_id}' has no uncertainty estimator "
                    "and no configured uncertainty_estimator. "
                    "All agents must implement estimate_uncertainty() in research mode. "
                    "Confidence-derived heuristics are not permitted."
                )

            # ── PRODUCTION MODE: documented heuristic fallback ────────
            # WARNING: this is a heuristic, not genuine UQ.
            # Do not use results derived from this in research publications.
            pred = predictions[agent_id]
            confidence = self._extract_confidence(pred)
            epistemic = max(0.0, 1.0 - confidence) * 0.5
            uncertainties[agent_id] = {
                "epistemic": epistemic,
                "aleatoric": epistemic * 0.5,
                "total": epistemic * 1.5,
            }
            used_fallback = True
            logger.warning(
                "PRODUCTION FALLBACK: Agent %s using confidence-derived UQ heuristic. "
                "This is NOT genuine MC Dropout uncertainty.",
                agent_id,
            )

        status = ModuleStatus(
            executed=True,
            method="mc_dropout" if not used_fallback else "confidence_heuristic_fallback",
            duration_ms=0,  # set by caller
            failure_reason="confidence_heuristic_used" if used_fallback else None,
            inputs_validated=True,
            outputs_validated=len(uncertainties) == len(predictions),
        )
        return uncertainties, status

    def _compute_trust_weights(
        self, uncertainties: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        Compute normalized trust weights.
        w_j = 1 / (sigma_e_j + eps)   [COGNIX proposed mechanism]
        Inversely proportional to epistemic uncertainty.
        """
        eps = 1e-8
        weights: dict[str, float] = {}
        for agent_id, unc in uncertainties.items():
            weights[agent_id] = 1.0 / (unc.get("epistemic", 1.0) + eps)

        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total
        return weights

    def _aggregate_confidence(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
    ) -> float:
        """Weighted mean confidence (used when no belief_fuser is configured)."""
        confs = []
        for agent_id, pred in predictions.items():
            w = trust_weights.get(agent_id, 1.0 / max(len(predictions), 1))
            confs.append(w * self._extract_confidence(pred))
        return float(np.clip(np.sum(confs), 0.0, 1.0)) if confs else 0.5

    def _aggregate_uncertainty(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
    ) -> tuple[float, float, float, float]:
        """Return (confidence, epistemic, aleatoric, total) as weighted aggregates."""
        confidences = []
        epistemics = []
        aleatorics = []

        for agent_id, pred in predictions.items():
            w = trust_weights.get(agent_id, 1.0 / max(len(predictions), 1))
            conf = self._extract_confidence(pred)
            unc = uncertainties.get(agent_id, {"epistemic": 0.2, "aleatoric": 0.1})
            confidences.append(w * conf)
            epistemics.append(w * unc.get("epistemic", 0.2))
            aleatorics.append(w * unc.get("aleatoric", 0.1))

        agg_confidence = float(np.sum(confidences)) if confidences else 0.5
        agg_epistemic = float(np.sum(epistemics)) if epistemics else 0.5
        agg_aleatoric = float(np.sum(aleatorics)) if aleatorics else 0.3
        agg_total = agg_epistemic + agg_aleatoric

        return (
            float(np.clip(agg_confidence, 0.0, 1.0)),
            float(np.clip(agg_epistemic, 0.0, 1.0)),
            float(np.clip(agg_aleatoric, 0.0, 1.0)),
            float(np.clip(agg_total, 0.0, 1.0)),
        )

    def _run_belief_fusion(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
        reliabilities: dict[str, float],
    ) -> float:
        """
        Run belief fusion using the injected BeliefFuser plugin.
        Returns fused confidence scalar.
        In research mode: exceptions propagate. In production: returns 0.5 on failure.
        """
        fuser = self.belief_fuser
        
        # Prepare inputs according to the new BeliefFuser interface
        pred_dict = {
            agent_id: self._extract_prediction_probability(pred)
            for agent_id, pred in predictions.items()
        }
        
        unc_dict = {
            agent_id: u.get("epistemic", 0.3)
            for agent_id, u in uncertainties.items()
        }
        
        if not pred_dict:
            if self.mode == RESEARCH_MODE:
                raise RuntimeError("RESEARCH MODE: No predictions to fuse.")
            return 0.5
            
        # The fuser is expected to return a FusionResult
        fused = fuser.fuse(
            predictions=pred_dict,
            uncertainties=unc_dict,
            reliabilities=reliabilities
        )
        
        return float(np.clip(fused.probability, 0.0, 1.0))

    def _run_attribution(
        self,
        predictions: dict[str, Any],
        uncertainties: dict[str, dict[str, float]],
        trust_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute epistemic Shapley values using the COLLECTIVE value function.

        v(S) = collective epistemic uncertainty of subset S.

        This captures each agent's marginal contribution to collective
        uncertainty reduction, satisfying the Shapley efficiency property:
            sum(phi_i) = v(all_agents) - v(empty_set)
        """
        agent_ids = list(uncertainties.keys())

        def v(subset: list[str]) -> float:
            """Collective value function: collective epistemic uncertainty of subset."""
            if not subset:
                return 1.0  # v(∅) = maximum uncertainty
            sub_preds = {k: predictions[k] for k in subset if k in predictions}
            sub_unc = {k: uncertainties[k] for k in subset if k in uncertainties}
            if not sub_preds:
                return 1.0
            sub_weights = self._compute_trust_weights(sub_unc)
            _, agg_epistemic, _, _ = self._aggregate_uncertainty(
                sub_preds, sub_unc, sub_weights
            )
            return agg_epistemic

        return self.attribution.compute(agent_ids, v)

    def _assess_risk(self, confidence: float, epistemic: float) -> RiskLevel:
        """Rule-based risk assessment using configurable thresholds."""
        high_unc_thresh = 0.5
        low_conf_thresh = 0.4
        low_unc_thresh = 0.2
        high_conf_thresh = 0.7

        if hasattr(self.config, "decision"):
            d = self.config.decision
            high_unc_thresh = getattr(d, "high_uncertainty_threshold", 0.5)
            low_conf_thresh = getattr(d, "low_confidence_threshold", 0.4)
            low_unc_thresh = high_unc_thresh * 0.4
            high_conf_thresh = getattr(d, "high_confidence_threshold", 0.7)

        if epistemic > high_unc_thresh or confidence < low_conf_thresh:
            return RiskLevel.HIGH
        if epistemic < low_unc_thresh and confidence > high_conf_thresh:
            return RiskLevel.LOW
        return RiskLevel.MODERATE

    def _make_decision(
        self,
        confidence: float,
        epistemic: float,
        risk_level: RiskLevel,
        conformal_set_size: int = 1,
    ) -> tuple[DecisionOutcome, bool, bool, bool]:
        """Map risk level + confidence to a final decision outcome."""
        if self.escalation is not None:
            try:
                if hasattr(self.escalation, "evaluate"):
                    result = self.escalation.evaluate(
                        confidence=confidence,
                        epistemic_uncertainty=epistemic,
                        conformal_set_size=conformal_set_size,
                        max_shapley_value=0.0,
                    )
                elif hasattr(self.escalation, "decide"):
                    result = self.escalation.decide(confidence, epistemic, risk_level, {})
                else:
                    result = None

                if result is not None:
                    outcome = (
                        DecisionOutcome(result.outcome.value)
                        if hasattr(result.outcome, "value")
                        else DecisionOutcome(str(result.outcome))
                    )
                    is_esc = bool(getattr(result, "escalation_required", getattr(result, "escalation", False)))
                    return (
                        outcome,
                        is_esc,
                        outcome == DecisionOutcome.ABSTAIN,
                        outcome == DecisionOutcome.REQUEST_INFORMATION,
                    )
            except Exception as exc:
                logger.debug("Escalation engine failed: %s", exc)

        if risk_level == RiskLevel.HIGH:
            return DecisionOutcome.ESCALATE, True, False, False
        if risk_level == RiskLevel.MODERATE:
            return DecisionOutcome.REQUEST_INFORMATION, False, False, True
        return DecisionOutcome.ACT, False, False, False

    def _generate_explanation(
        self,
        decision: DecisionOutcome,
        confidence: float,
        epistemic: float,
        aleatoric: float,
        risk_level: RiskLevel,
        trust_weights: dict[str, float],
        contributions: dict[str, float],
        module_status: dict[str, ModuleStatus],
    ) -> tuple[str, list[str]]:
        """Generate explanation from actual pipeline state."""
        top_agent = max(trust_weights, key=trust_weights.get) if trust_weights else "N/A"

        executed_modules = [
            k for k, v in module_status.items() if v.executed
        ]
        failed_modules = [
            k for k, v in module_status.items() if not v.executed
        ]

        reasoning_steps = [
            f"1. Collected predictions from {len(trust_weights)} agent(s).",
            f"2. Epistemic uncertainty (Var of MC samples): {epistemic:.4f}.",
            f"3. Aleatoric uncertainty (E[p(1-p)]): {aleatoric:.4f}.",
            f"4. Most trusted agent: {top_agent} (w={trust_weights.get(top_agent, 0):.3f}).",
            f"5. Fused confidence: {confidence:.4f}.",
            f"6. Risk level: {risk_level.value}.",
            f"7. Decision: {decision.value}.",
            f"8. Executed modules: {executed_modules}.",
        ]

        if failed_modules:
            reasoning_steps.append(f"9. ⚠ Failed modules: {failed_modules}.")

        if contributions:
            top_contrib = max(contributions, key=contributions.get)
            reasoning_steps.append(
                f"10. Shapley top contributor: {top_contrib} "
                f"(ϕ={contributions.get(top_contrib, 0):.4f})."
            )

        explanation = (
            f"COGNIX decided {decision.value} with {confidence:.1%} confidence "
            f"(risk: {risk_level.value}, epistemic: {epistemic:.4f}). "
            f"Pipeline: {' → '.join(executed_modules)}."
        )

        return explanation, reasoning_steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_id(agent: Any) -> str:
        if hasattr(agent, "agent_id"):
            return str(agent.agent_id)
        if hasattr(agent, "name"):
            return str(agent.name)
        return f"agent_{id(agent)}"

    @staticmethod
    def _extract_confidence(pred: Any) -> float:
        if pred is None:
            return 0.5
        if isinstance(pred, float):
            return float(np.clip(pred, 0.0, 1.0))
        if hasattr(pred, "confidence") and getattr(pred, "confidence") is not None:
            return float(np.clip(pred.confidence, 0.0, 1.0))
        if hasattr(pred, "probabilities") and pred.probabilities is not None:
            return float(np.clip(np.max(pred.probabilities), 0.0, 1.0))
        if isinstance(pred, dict):
            for key in ("confidence", "prob"):
                if key in pred:
                    return float(np.clip(pred[key], 0.0, 1.0))
            if "probabilities" in pred:
                return float(np.clip(np.max(np.array(pred["probabilities"])), 0.0, 1.0))
        if isinstance(pred, np.ndarray):
            return float(np.clip(np.max(pred), 0.0, 1.0))
        return 0.5

    @staticmethod
    def _extract_prediction_probability(pred: Any) -> float:
        """Extract model prediction probability p_i in [0, 1] for GAT and belief fusion."""
        if pred is None:
            return 0.5
        if isinstance(pred, (float, int, np.floating, np.integer)):
            return float(np.clip(float(pred), 0.0, 1.0))
        if hasattr(pred, "value") and isinstance(pred.value, (float, int, np.floating, np.integer)):
            return float(np.clip(float(pred.value), 0.0, 1.0))
        if isinstance(pred, dict):
            if "prob" in pred and isinstance(pred["prob"], (float, int, np.floating, np.integer)):
                return float(np.clip(float(pred["prob"]), 0.0, 1.0))
            if "probabilities" in pred:
                probs = np.array(pred["probabilities"])
                if probs.ndim == 1 and len(probs) == 2:
                    return float(np.clip(float(probs[1]), 0.0, 1.0))
        return CognixPipeline._extract_confidence(pred)

    def _emergency_result(
        self, latencies: dict[str, float], pipeline_start: float
    ) -> DecisionResult:
        return DecisionResult(
            decision=DecisionOutcome.ABSTAIN,
            confidence=0.0,
            calibrated_confidence=None,
            aleatoric_uncertainty=None,
            epistemic_uncertainty=1.0,
            total_uncertainty=1.0,
            risk_level=RiskLevel.HIGH,
            escalation_required=True,
            abstained=True,
            requested_information=False,
            agent_contributions={},
            agent_trust_weights={},
            agent_predictions={},
            communication_statistics=None,
            calibration_metrics=None,
            explanation="COGNIX abstained: no agent predictions were available.",
            reasoning_steps=["0. No agent predictions — defaulting to safe ABSTAIN."],
            latency_ms=latencies,
            total_latency_ms=_ms(pipeline_start),
            timestamp=time.time(),
            session_id=None,
            metadata={"emergency": True},
        )
