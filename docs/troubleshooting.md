# 🛠️ Troubleshooting & Common Issues

This guide details common diagnostic errors, root causes, and verified solutions when installing, configuring, or running COGNIX v0.1.0.

---

## 1. Installation & Environment Issues

### Issue: `pip install cognix` fails or installs the wrong package
- **Symptom**: `ERROR: Could not find a version that satisfies the requirement cognix` or an unrelated legacy package is downloaded.
- **Root Cause**: The official distribution name on PyPI is **`cognix-uncertainty-framework`**, not `cognix`.
- **Solution**:
  ```bash
  python -m pip install cognix-uncertainty-framework==0.1.0
  ```

---

### Issue: Package installed but `import cognix` fails on Windows
- **Symptom**: `ModuleNotFoundError: No module named 'cognix'` after running `pip install cognix-uncertainty-framework`.
- **Root Cause**: Multiple Python interpreters are installed on Windows (e.g. Python 3.14 and Python 3.12). The `pip` command ran on one interpreter, while `python` launched a different one.
- **Diagnostic**:
  ```powershell
  py -0p
  ```
- **Solution**:
  Target the specific interpreter (Python 3.12) or create a virtual environment:
  ```powershell
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install cognix-uncertainty-framework==0.1.0
  ```

---

### Issue: Source Tree Leakage During Development
- **Symptom**: Changes to your installed package have no effect, or imports resolve to the local Git repository instead of `site-packages`.
- **Root Cause**: Python's `sys.path[0]` defaults to the current working directory. Running commands from within the cloned `Cognix_` repository imports the local directory rather than the installed wheel.
- **Diagnostic**:
  ```bash
  python -c "import cognix; print(cognix.__file__)"
  ```
- **Solution**:
  Test outside the repository directory, or install in editable mode:
  ```bash
  python -m pip install -e .
  ```

---

## 2. Dependency & Extras Errors

### Issue: `ModuleNotFoundError: No module named 'torch'`
- **Symptom**:
  ```text
  ImportError: Graph module 'epistemic_gat' requires PyTorch. Install with: pip install 'cognix-uncertainty-framework[graph]'
  ```
  or `No module named 'torch'` when calling `MonteCarloDropout.estimate()`, `DeepEnsemble.estimate()`, or `StandardAgent.predict()`.
- **Root Cause**: Base COGNIX is pure Python/NumPy. PyTorch is an optional extra.
- **Solution**:
  Install the `[graph]` or `[vision]` extra:
  ```bash
  python -m pip install "cognix-uncertainty-framework[graph]==0.1.0"
  ```
  *Alternative*: For pure Python / NumPy environments, use `UncertaintyDecomposition`, `EpistemicWeightedFusion`, and custom agent classes without PyTorch.

---

### Issue: `ImportError: scipy is required for Temperature Scaling optimization`
- **Symptom**: `ImportError` when calling `TemperatureScaling.fit()`.
- **Root Cause**: SciPy is missing from your active environment.
- **Solution**:
  ```bash
  python -m pip install "scipy>=1.10"
  ```

---

## 3. Runtime & Pipeline Execution Errors

### Issue: `RuntimeError: RESEARCH MODE: Agent has no uncertainty estimator`
- **Symptom**:
  ```text
  RuntimeError: RESEARCH MODE: Agent 'camera' has no uncertainty estimator and no configured uncertainty_estimator. All agents must implement estimate_uncertainty() in research mode.
  ```
- **Root Cause**: The engine was instantiated with `mode="research"`. Research mode strictly forbids confidence-derived heuristic fallbacks.
- **Solution**:
  Either:
  1. Implement `estimate_uncertainty(observation)` on all agents returning `{"epistemic": float, "aleatoric": float}`.
  2. Switch to production mode: `engine = DecisionEngine(..., mode="production")`.

---

### Issue: `RuntimeError: Conformal predictor must be calibrated before prediction.`
- **Symptom**: Exception raised during `engine.decide()` or `calibrator.predict()`.
- **Root Cause**: `ConformalPredictor` requires calibration data before it can produce prediction sets.
- **Solution**:
  Fit the calibrator on validation probabilities and labels before inference:
  ```python
  calibrator = ConformalPredictor()
  calibrator.fit(cal_outputs=val_probs, cal_labels=val_labels)

  engine = DecisionEngine(..., calibrator=calibrator)
  ```

---

### Issue: `ValueError: low_confidence_threshold (0.7) must be < high_confidence_threshold (0.4)`
- **Symptom**: Pydantic validation fails during `CognixConfig` initialization or YAML loading.
- **Root Cause**: `DecisionConfig.low_confidence_threshold` is greater than or equal to `high_confidence_threshold`.
- **Solution**:
  Ensure thresholds satisfy the strict inequality:
  ```python
  from cognix import CognixConfig
  from cognix.config.schema import DecisionConfig

  config = CognixConfig(
      decision=DecisionConfig(
          low_confidence_threshold=0.40,
          high_confidence_threshold=0.70,
      )
  )
  ```

---

### Issue: `ValueError: Non-finite prediction value: nan`
- **Symptom**: Pipeline rejects agent output with `ValueError`.
- **Root Cause**: An agent model produced `NaN` or `Inf` (e.g. from numerical overflow, zero division, or exploding gradients).
- **Solution**:
  Validate agent models before inference. COGNIX enforces strict finite-number checks across all predictions and uncertainties.

---

### Issue: `EscalationEngine.evaluate()` outcome comparison fails identity check
- **Symptom**:
  ```python
  res = escalation_engine.evaluate(0.85, 0.1)
  assert res.outcome == DecisionOutcome.ACT  # Fails!
  ```
- **Root Cause**: `cognix.DecisionOutcome` is imported from `cognix.engine.result`, whereas `EscalationEngine` defines its own matching enum in `cognix.decision.escalation`. While identical in string values, they are separate Enum classes in Python.
- **Solution**:
  Compare via string value:
  ```python
  assert res.outcome.value == DecisionOutcome.ACT.value
  ```
