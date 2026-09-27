import numpy as np
from datetime import datetime
from typing import Any, Callable, Optional, Dict
from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult


class CallableAgentAdapter(AgentInterface):
    """
    Wraps any Python callable as a standard COGNIX agent conforming to AgentInterface.
    """

    def __init__(
        self,
        agent_id: str,
        func: Callable[[Any], Any],
        name: str = "CallableAgent",
        uncertainty_fn: Optional[Callable[[Any], UncertaintyResult]] = None,
        description: str = "Callable wrapper",
    ):
        self.agent_id = agent_id
        self.func = func
        self.name = name
        self.uncertainty_fn = uncertainty_fn
        self.description = description
        self.healthy = True

    def predict(self, inputs: Any) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(
                value=0.5,
                confidence=0.0,
                metadata={"agent_id": self.agent_id, "status": "unhealthy"},
            )

        try:
            result = self.func(inputs)
        except Exception as exc:
            raise ValueError(
                f"CallableAgentAdapter '{self.agent_id}' execution failed: {exc}"
            ) from exc

        if isinstance(result, PredictionResult):
            return result

        # Extract value and confidence from result
        if isinstance(result, dict):
            raw_val = result.get("prediction", result.get("value", result.get("output")))
            conf = result.get("confidence")
        elif (
            isinstance(result, (tuple, list))
            and len(result) == 2
            and isinstance(result[1], (float, int))
        ):
            raw_val, conf = result[0], float(result[1])
        else:
            raw_val = result
            conf = None

        if isinstance(raw_val, (int, float, np.floating, np.integer)):
            if np.isnan(raw_val) or np.isinf(raw_val):
                raise ValueError(f"Callable output is NaN or Inf: {raw_val}")
            val = float(raw_val)
            if conf is None:
                conf = float(val) if (0.0 <= val <= 1.0) else 1.0
        elif isinstance(raw_val, np.ndarray):
            if np.any(np.isnan(raw_val)) or np.any(np.isinf(raw_val)):
                raise ValueError("Callable output contains NaN or Inf")
            if raw_val.size == 1:
                val = float(raw_val.item())
                if conf is None:
                    conf = float(val) if (0.0 <= val <= 1.0) else 1.0
            elif (
                raw_val.ndim == 1
                and len(raw_val) == 2
                and np.isclose(np.sum(raw_val), 1.0, atol=1e-2)
            ):
                val = float(raw_val[1])
                if conf is None:
                    conf = float(np.max(raw_val))
            else:
                val = raw_val
                if conf is None:
                    conf = 1.0
        else:
            val = raw_val
            if conf is None:
                conf = 1.0

        if conf is not None:
            if np.isnan(conf) or np.isinf(conf):
                raise ValueError(f"Confidence is NaN or Inf: {conf}")
            conf = float(np.clip(conf, 0.0, 1.0))

        return PredictionResult(
            value=val,
            confidence=conf,
            metadata={
                "agent_id": self.agent_id,
                "name": self.name,
                "raw_output": result,
            },
        )

    def estimate_uncertainty(self, inputs: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(
                prediction=0.5,
                epistemic=1.0,
                aleatoric=1.0,
                total=2.0,
            )

        if self.uncertainty_fn is not None:
            est = self.uncertainty_fn(inputs)
            if isinstance(est, UncertaintyResult):
                return est
            if isinstance(est, dict):
                return UncertaintyResult(
                    prediction=float(est["prediction"]),
                    epistemic=float(est["epistemic"]),
                    aleatoric=float(est["aleatoric"]),
                    total=float(est.get("total", est["epistemic"] + est["aleatoric"])),
                )
            raise ValueError(f"uncertainty_fn returned unexpected type: {type(est)}")

        pred_res = self.predict(inputs)
        val = pred_res.value
        if isinstance(val, (int, float, np.floating, np.integer)):
            p = float(val)
        elif isinstance(val, np.ndarray) and val.size == 1:
            p = float(val.item())
        else:
            p = float(pred_res.confidence) if pred_res.confidence is not None else 0.5

        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Cannot estimate uncertainty for unnormalized value {p}; expected in [0, 1]"
            )

        # External deterministic callables cannot measure epistemic variance.
        # Epistemic uncertainty is honestly 0.0 (non-fabricated).
        # Aleatoric data uncertainty for Bernoulli parameter p is p(1-p).
        aleatoric = float(p * (1.0 - p))
        return UncertaintyResult(
            prediction=p,
            epistemic=0.0,
            aleatoric=aleatoric,
            total=aleatoric,
        )

    def metadata(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "callable",
            "description": self.description,
            "healthy": self.healthy,
        }

    def health(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "is_healthy": self.healthy,
        }

    def explain(self, inputs: Any) -> str:
        func_name = getattr(self.func, "__name__", str(self.func))
        return f"Prediction generated by wrapped callable {func_name}."


class SklearnAgentAdapter(AgentInterface):
    """
    Wraps scikit-learn models implementing predict or predict_proba.
    """

    def __init__(
        self,
        agent_id: str,
        model: Any,
        name: str = "SklearnAgent",
        uncertainty_estimator: Optional[Any] = None,
    ):
        self.agent_id = agent_id
        self.model = model
        self.name = name
        self.uncertainty_estimator = uncertainty_estimator
        self.healthy = True

    def predict(self, inputs: Any) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(
                value=0.5,
                confidence=0.0,
                metadata={"agent_id": self.agent_id, "status": "unhealthy"},
            )

        inputs_arr = np.asarray(inputs)
        if inputs_arr.ndim == 1:
            inputs_arr = inputs_arr.reshape(1, -1)
        elif inputs_arr.ndim == 0:
            inputs_arr = inputs_arr.reshape(1, -1)

        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(inputs_arr)
                if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                    raise ValueError("Model predict_proba returned NaN or Inf")
                if probs.shape[1] == 2:
                    val = float(probs[0, 1])
                    conf = float(np.max(probs[0]))
                else:
                    pred = self.model.predict(inputs_arr)
                    val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                    conf = float(np.max(probs[0]))
                raw_out = probs
            else:
                pred = self.model.predict(inputs_arr)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    raise ValueError("Model predict returned NaN or Inf")
                val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                conf = 1.0 if not (0.0 <= val <= 1.0) else max(val, 1.0 - val)
                raw_out = pred
        except Exception as exc:
            raise ValueError(
                f"SklearnAgentAdapter '{self.agent_id}' prediction failed: {exc}"
            ) from exc

        return PredictionResult(
            value=val,
            confidence=conf,
            metadata={
                "agent_id": self.agent_id,
                "name": self.name,
                "raw_output": raw_out,
            },
        )

    def estimate_uncertainty(self, inputs: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(
                prediction=0.5,
                epistemic=1.0,
                aleatoric=1.0,
                total=2.0,
            )

        if self.uncertainty_estimator is not None:
            return self.uncertainty_estimator.estimate(self.model, inputs)

        inputs_arr = np.asarray(inputs)
        if inputs_arr.ndim == 1:
            inputs_arr = inputs_arr.reshape(1, -1)
        elif inputs_arr.ndim == 0:
            inputs_arr = inputs_arr.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(inputs_arr)[0]
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                raise ValueError("Model predict_proba returned NaN or Inf")
            if len(probs) == 2:
                p = float(np.clip(probs[1], 0.0, 1.0))
                aleatoric = float(p * (1.0 - p))
            else:
                p = float(np.clip(np.max(probs), 0.0, 1.0))
                aleatoric = float(
                    -np.sum(probs * np.log(probs + 1e-12))
                    / np.log(len(probs) + 1e-12)
                )
        else:
            pred = self.model.predict(inputs_arr)
            raw = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            if not (0.0 <= raw <= 1.0):
                raise ValueError(
                    f"Cannot estimate uncertainty for non-probabilistic prediction {raw}"
                )
            p = raw
            aleatoric = 0.0

        # A standard single fitted sklearn model cannot provide epistemic uncertainty.
        # Epistemic uncertainty is set to 0.0 (non-fabricated).
        return UncertaintyResult(
            prediction=p,
            epistemic=0.0,
            aleatoric=aleatoric,
            total=aleatoric,
        )

    def metadata(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "sklearn",
            "model_type": type(self.model).__name__,
            "healthy": self.healthy,
        }

    def health(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "is_healthy": self.healthy,
        }

    def explain(self, inputs: Any) -> str:
        return f"Prediction generated by sklearn model {type(self.model).__name__}."


class PyTorchAgentAdapter(AgentInterface):
    """
    Wraps PyTorch nn.Module models.
    Supports single-pass deterministic inference and multi-pass MC Dropout.
    """

    def __init__(
        self,
        agent_id: str,
        model: Any,
        name: str = "PyTorchAgent",
        num_mc_samples: int = 1,
        uncertainty_estimator: Optional[Any] = None,
    ):
        self.agent_id = agent_id
        self.model = model
        self.name = name
        self.num_mc_samples = num_mc_samples
        self.uncertainty_estimator = uncertainty_estimator
        self.healthy = True
        import torch
        self.torch = torch

    def predict(self, inputs: Any) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(
                value=0.5,
                confidence=0.0,
                metadata={"agent_id": self.agent_id, "status": "unhealthy"},
            )

        self.model.eval()
        t_in = (
            inputs
            if isinstance(inputs, self.torch.Tensor)
            else self.torch.as_tensor(inputs, dtype=self.torch.float32)
        )
        try:
            with self.torch.no_grad():
                outputs = self.model(t_in)
        except Exception as exc:
            raise ValueError(
                f"PyTorchAgentAdapter '{self.agent_id}' execution failed: {exc}"
            ) from exc

        if not isinstance(outputs, self.torch.Tensor):
            raise ValueError(
                f"Expected torch.Tensor from PyTorch model, got {type(outputs)}"
            )

        if self.torch.isnan(outputs).any() or self.torch.isinf(outputs).any():
            raise ValueError("PyTorch model output contains NaN or Inf")

        out_flat = outputs.squeeze()
        if out_flat.dim() == 0 or (out_flat.dim() == 1 and out_flat.numel() == 1):
            val = float(out_flat.item())
            if 0.0 <= val <= 1.0:
                conf = max(val, 1.0 - val)
            else:
                prob = float(self.torch.sigmoid(out_flat).item())
                val = prob
                conf = max(val, 1.0 - val)
        elif out_flat.dim() == 1 and out_flat.numel() == 2:
            probs = self.torch.softmax(out_flat, dim=-1).cpu().numpy()
            val = float(probs[1])
            conf = float(np.max(probs))
        elif out_flat.dim() == 1:
            probs = self.torch.softmax(out_flat, dim=-1).cpu().numpy()
            val = float(np.argmax(probs))
            conf = float(np.max(probs))
        else:
            probs = self.torch.softmax(out_flat, dim=-1).cpu().numpy()
            val = float(np.argmax(probs[0]))
            conf = float(np.max(probs[0]))

        return PredictionResult(
            value=val,
            confidence=conf,
            metadata={
                "agent_id": self.agent_id,
                "name": self.name,
                "raw_output": outputs,
            },
        )

    def estimate_uncertainty(self, inputs: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(
                prediction=0.5,
                epistemic=1.0,
                aleatoric=1.0,
                total=2.0,
            )

        if self.uncertainty_estimator is not None:
            return self.uncertainty_estimator.estimate(self.model, inputs)

        t_in = (
            inputs
            if isinstance(inputs, self.torch.Tensor)
            else self.torch.as_tensor(inputs, dtype=self.torch.float32)
        )

        if self.num_mc_samples > 1:
            # Genuine MC Dropout estimation over multiple stochastic forward passes
            self.model.train()
            samples = []
            with self.torch.no_grad():
                for _ in range(self.num_mc_samples):
                    out = self.model(t_in)
                    if self.torch.isnan(out).any() or self.torch.isinf(out).any():
                        raise ValueError(
                            "PyTorch model output contains NaN or Inf during MC sampling"
                        )
                    out_flat = out.squeeze()
                    if out_flat.dim() == 0 or (
                        out_flat.dim() == 1 and out_flat.numel() == 1
                    ):
                        p = float(out_flat.item())
                        if not (0.0 <= p <= 1.0):
                            p = float(self.torch.sigmoid(out_flat).item())
                    elif out_flat.dim() == 1 and out_flat.numel() == 2:
                        p = float(self.torch.softmax(out_flat, dim=-1)[1].item())
                    else:
                        p = float(self.torch.softmax(out_flat, dim=-1)[0].item())
                    samples.append(p)

            samples_arr = np.array(samples)
            p_bar = float(np.mean(samples_arr))
            epi = float(np.mean((samples_arr - p_bar) ** 2))
            ale = float(np.mean(samples_arr * (1.0 - samples_arr)))
            p_bar = float(np.clip(p_bar, 0.0, 1.0))
            return UncertaintyResult(
                prediction=p_bar,
                epistemic=epi,
                aleatoric=ale,
                total=epi + ale,
            )
        else:
            # Deterministic single pass: epistemic uncertainty is 0.0 (non-fabricated)
            pred_res = self.predict(inputs)
            val = pred_res.value
            if not isinstance(val, (int, float, np.floating, np.integer)) or not (
                0.0 <= float(val) <= 1.0
            ):
                raise ValueError(
                    f"Cannot estimate uncertainty: prediction {val} must be in [0, 1]"
                )
            p = float(val)
            aleatoric = float(p * (1.0 - p))
            return UncertaintyResult(
                prediction=p,
                epistemic=0.0,
                aleatoric=aleatoric,
                total=aleatoric,
            )

    def metadata(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "pytorch",
            "model_type": type(self.model).__name__,
            "mc_samples": self.num_mc_samples,
            "healthy": self.healthy,
        }

    def health(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "is_healthy": self.healthy,
        }

    def explain(self, inputs: Any) -> str:
        return f"Prediction generated by PyTorch model {type(self.model).__name__}."


class LLMAgentAdapter(AgentInterface):
    """
    Wraps an LLM callable that returns text and optional confidence/probability.
    """

    def __init__(
        self,
        agent_id: str,
        llm_callable: Callable[[Any], Any],
        name: str = "LLMAgent",
        uncertainty_fn: Optional[Callable[[Any], UncertaintyResult]] = None,
    ):
        self.agent_id = agent_id
        self.llm_callable = llm_callable
        self.name = name
        self.uncertainty_fn = uncertainty_fn
        self.healthy = True

    def predict(self, inputs: Any) -> PredictionResult:
        if not self.healthy:
            return PredictionResult(
                value=0.5,
                confidence=0.0,
                metadata={"agent_id": self.agent_id, "status": "unhealthy"},
            )

        try:
            result = self.llm_callable(inputs)
        except Exception as exc:
            raise ValueError(
                f"LLMAgentAdapter '{self.agent_id}' execution failed: {exc}"
            ) from exc

        if isinstance(result, PredictionResult):
            return result

        if isinstance(result, dict):
            text = str(result.get("text", result.get("response", result)))
            conf = result.get("confidence", 0.8)
            val = result.get("probability", result.get("value", text))
        else:
            text = str(result)
            conf = 0.8
            val = text

        if conf is not None:
            if np.isnan(conf) or np.isinf(conf):
                raise ValueError(f"LLM confidence is NaN or Inf: {conf}")
            conf = float(np.clip(float(conf), 0.0, 1.0))

        return PredictionResult(
            value=val,
            confidence=conf,
            metadata={
                "agent_id": self.agent_id,
                "name": self.name,
                "text": text,
                "raw_output": result,
            },
        )

    def estimate_uncertainty(self, inputs: Any) -> UncertaintyResult:
        if not self.healthy:
            return UncertaintyResult(
                prediction=0.5,
                epistemic=1.0,
                aleatoric=1.0,
                total=2.0,
            )

        if self.uncertainty_fn is not None:
            est = self.uncertainty_fn(inputs)
            if isinstance(est, UncertaintyResult):
                return est
            if isinstance(est, dict):
                return UncertaintyResult(
                    prediction=float(est["prediction"]),
                    epistemic=float(est["epistemic"]),
                    aleatoric=float(est["aleatoric"]),
                    total=float(est.get("total", est["epistemic"] + est["aleatoric"])),
                )
            raise ValueError(f"uncertainty_fn returned unexpected type: {type(est)}")

        pred = self.predict(inputs)
        val = pred.value
        if (
            isinstance(val, (int, float, np.floating, np.integer))
            and 0.0 <= float(val) <= 1.0
        ):
            p = float(val)
        elif pred.confidence is not None:
            p = float(pred.confidence)
        else:
            p = 0.5

        # Single LLM calls cannot measure epistemic uncertainty without multi-sample generations.
        # Epistemic uncertainty is honestly reported as 0.0 (non-fabricated).
        aleatoric = float(p * (1.0 - p))
        return UncertaintyResult(
            prediction=p,
            epistemic=0.0,
            aleatoric=aleatoric,
            total=aleatoric,
        )

    def metadata(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "llm",
            "healthy": self.healthy,
        }

    def health(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "healthy": self.healthy,
            "is_healthy": self.healthy,
        }

    def explain(self, inputs: Any) -> str:
        return "Prediction generated by language model."
