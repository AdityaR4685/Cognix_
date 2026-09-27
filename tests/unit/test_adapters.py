import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from cognix.core.interfaces import AgentInterface
from cognix.core.types import PredictionResult, UncertaintyResult
from cognix.agents.adapters import (
    CallableAgentAdapter,
    SklearnAgentAdapter,
    PyTorchAgentAdapter,
    LLMAgentAdapter,
)
from cognix.engine.decision_engine import DecisionEngine
from cognix.engine.result import DecisionResult


# ── 1. Module import & Interface Conformance ──────────────────────────────────

def test_adapter_module_import():
    """Verify adapters import cleanly and implement AgentInterface."""
    for cls in [CallableAgentAdapter, SklearnAgentAdapter, PyTorchAgentAdapter, LLMAgentAdapter]:
        assert issubclass(cls, AgentInterface)


# ── 2. CallableAgentAdapter ───────────────────────────────────────────────────

def test_callable_adapter_basic_prediction():
    """Verify CallableAgentAdapter wraps a simple callable returning probability."""
    def dummy_classifier(x):
        return 0.85

    adapter = CallableAgentAdapter("callable_1", dummy_classifier, name="DummyClassifier")
    pred = adapter.predict([1.0, 2.0])

    assert isinstance(pred, PredictionResult)
    assert np.isclose(pred.value, 0.85)
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.metadata["agent_id"] == "callable_1"
    assert pred.metadata["name"] == "DummyClassifier"


def test_callable_adapter_uncertainty_contract():
    """Verify uncertainty output contract and non-fabricated epistemic uncertainty."""
    adapter = CallableAgentAdapter("callable_1", lambda x: 0.8)
    unc = adapter.estimate_uncertainty([1.0, 2.0])

    assert isinstance(unc, UncertaintyResult)
    assert np.isclose(unc.prediction, 0.8)
    # External deterministic callable must not fabricate epistemic uncertainty
    assert unc.epistemic == 0.0
    # Aleatoric uncertainty should match p * (1 - p)
    assert np.isclose(unc.aleatoric, 0.8 * 0.2)
    assert np.isclose(unc.total, unc.epistemic + unc.aleatoric)
    assert 0.0 <= unc.prediction <= 1.0


def test_callable_adapter_custom_uncertainty_fn():
    """Verify user can inject an explicit uncertainty function."""
    def custom_uq(x):
        return UncertaintyResult(prediction=0.7, epistemic=0.05, aleatoric=0.15, total=0.20)

    adapter = CallableAgentAdapter("callable_custom", lambda x: 0.7, uncertainty_fn=custom_uq)
    unc = adapter.estimate_uncertainty([1.0])
    assert unc.epistemic == 0.05
    assert unc.aleatoric == 0.15


# ── 3. SklearnAgentAdapter ────────────────────────────────────────────────────

def test_sklearn_adapter_fitted_classifier():
    """Verify SklearnAgentAdapter with a tiny fitted LogisticRegression model."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    adapter = SklearnAgentAdapter("sklearn_logreg", clf)

    # Prediction check
    pred = adapter.predict([0.5, 0.5])
    assert isinstance(pred, PredictionResult)
    assert isinstance(pred.value, float)
    assert 0.0 <= pred.value <= 1.0
    assert 0.0 <= pred.confidence <= 1.0

    # Uncertainty check
    unc = adapter.estimate_uncertainty([0.5, 0.5])
    assert isinstance(unc, UncertaintyResult)
    assert 0.0 <= unc.prediction <= 1.0
    # Single fitted sklearn model must not fabricate epistemic uncertainty
    assert unc.epistemic == 0.0
    assert unc.aleatoric >= 0.0
    assert unc.total >= 0.0


def test_sklearn_adapter_non_probabilistic_model():
    """Verify SklearnAgentAdapter gracefully handles models without predict_proba."""
    class DecisionBoundaryModel:
        def predict(self, X):
            return np.array([1.0])

    adapter = SklearnAgentAdapter("sklearn_boundary", DecisionBoundaryModel())
    pred = adapter.predict([1.0, 2.0])
    assert isinstance(pred, PredictionResult)
    assert pred.value == 1.0

    unc = adapter.estimate_uncertainty([1.0, 2.0])
    assert isinstance(unc, UncertaintyResult)
    assert unc.prediction == 1.0
    assert unc.epistemic == 0.0


# ── 4. PyTorchAgentAdapter ────────────────────────────────────────────────────

def test_pytorch_adapter_deterministic_model():
    """Verify PyTorchAgentAdapter with a tiny deterministic model."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid(),
    )
    adapter = PyTorchAgentAdapter("pytorch_det", model, num_mc_samples=1)

    x = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    pred = adapter.predict(x)

    assert isinstance(pred, PredictionResult)
    assert 0.0 <= pred.value <= 1.0
    assert 0.0 <= pred.confidence <= 1.0

    unc = adapter.estimate_uncertainty(x)
    assert isinstance(unc, UncertaintyResult)
    assert 0.0 <= unc.prediction <= 1.0
    assert unc.epistemic == 0.0  # Single pass: no epistemic fabrication
    assert np.isclose(unc.aleatoric, unc.prediction * (1.0 - unc.prediction))
    assert np.isclose(unc.total, unc.aleatoric)


def test_pytorch_adapter_mc_dropout():
    """Verify PyTorchAgentAdapter performs genuine MC Dropout when num_mc_samples > 1."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )
    adapter = PyTorchAgentAdapter("pytorch_mc", model, num_mc_samples=20)

    x = np.array([0.5, 0.1, -0.3, 0.8], dtype=np.float32)
    unc = adapter.estimate_uncertainty(x)

    assert isinstance(unc, UncertaintyResult)
    assert 0.0 <= unc.prediction <= 1.0
    assert unc.epistemic >= 0.0
    assert unc.aleatoric >= 0.0
    assert np.isclose(unc.total, unc.epistemic + unc.aleatoric)


# ── 5. LLMAgentAdapter ────────────────────────────────────────────────────────

def test_llm_adapter_dict_response():
    """Verify LLMAgentAdapter with mocked local callable returning dict."""
    def mock_llm_callable(prompt):
        return {
            "text": "Hazard detected ahead",
            "confidence": 0.85,
            "probability": 0.85,
        }

    adapter = LLMAgentAdapter("llm_agent_1", mock_llm_callable)
    pred = adapter.predict("Current camera frame analysis")

    assert isinstance(pred, PredictionResult)
    assert np.isclose(pred.confidence, 0.85)
    assert pred.metadata["text"] == "Hazard detected ahead"

    unc = adapter.estimate_uncertainty("Current camera frame analysis")
    assert isinstance(unc, UncertaintyResult)
    assert np.isclose(unc.prediction, 0.85)
    assert unc.epistemic == 0.0  # Honest: single LLM call cannot fabricate epistemic uncertainty
    assert np.isclose(unc.aleatoric, 0.85 * 0.15)
    assert np.isclose(unc.total, unc.aleatoric)


def test_llm_adapter_text_response():
    """Verify LLMAgentAdapter with mocked local callable returning plain text string."""
    adapter = LLMAgentAdapter("llm_agent_text", lambda prompt: "Clear road ahead")
    pred = adapter.predict("Input sensor description")

    assert isinstance(pred, PredictionResult)
    assert pred.value == "Clear road ahead"
    assert pred.confidence == 0.8

    unc = adapter.estimate_uncertainty("Input sensor description")
    assert isinstance(unc, UncertaintyResult)
    assert unc.epistemic == 0.0
    assert 0.0 <= unc.prediction <= 1.0


# ── 6. Malformed Output & Edge Cases ──────────────────────────────────────────

def test_callable_adapter_nan_fails_clearly():
    """Verify malformed NaN output from callable raises ValueError."""
    adapter = CallableAgentAdapter("bad_callable", lambda x: np.nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.predict([1.0])


def test_sklearn_adapter_nan_fails_clearly():
    """Verify NaN from model raises ValueError."""
    class NanModel:
        def predict_proba(self, X):
            return np.array([[np.nan, np.nan]])

    adapter = SklearnAgentAdapter("bad_sklearn", NanModel())
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.predict([1.0])


def test_pytorch_adapter_nan_fails_clearly():
    """Verify NaN tensor raises ValueError."""
    class NanModule(nn.Module):
        def forward(self, x):
            return torch.tensor([float("nan")])

    adapter = PyTorchAgentAdapter("bad_torch", NanModule())
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.predict([1.0])


def test_adapter_unhealthy_state():
    """Verify unhealthy agent produces canonical degraded fallback."""
    adapter = CallableAgentAdapter("unhealthy_agent", lambda x: 0.9)
    adapter.healthy = False

    pred = adapter.predict([1.0])
    assert pred.value == 0.5
    assert pred.confidence == 0.0
    assert pred.metadata.get("status") == "unhealthy"

    unc = adapter.estimate_uncertainty([1.0])
    assert unc.prediction == 0.5
    assert unc.epistemic == 1.0
    assert unc.aleatoric == 1.0
    assert unc.total == 2.0


# ── 7. Pipeline Integration ───────────────────────────────────────────────────

def test_adapters_in_decision_engine():
    """Verify all repaired adapters can run together seamlessly inside Cognix DecisionEngine."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([0, 1, 0, 1])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    torch_model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    agent1 = CallableAgentAdapter("agent_callable", lambda x: 0.75)
    agent2 = SklearnAgentAdapter("agent_sklearn", clf)
    agent3 = PyTorchAgentAdapter("agent_torch", torch_model)
    agent4 = LLMAgentAdapter("agent_llm", lambda x: {"text": "Safe", "probability": 0.7, "confidence": 0.7})

    engine = DecisionEngine(mode="production")
    result = engine.decide(
        agents=[agent1, agent2, agent3, agent4],
        input_data=np.array([0.5, 0.5], dtype=np.float32),
    )

    assert isinstance(result, DecisionResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.epistemic_uncertainty >= 0.0
    assert result.aleatoric_uncertainty >= 0.0
    assert result.decision is not None
