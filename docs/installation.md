# 📦 Installation Guide

This guide covers installing and verifying the official `cognix-uncertainty-framework` Python package on Windows, macOS, and Linux.

---

## Package Identification

| Field | Value |
|---|---|
| **PyPI Distribution Name** | `cognix-uncertainty-framework` |
| **Python Import Name** | `cognix` |
| **Current Verified Version** | `0.1.0` |
| **Public Export Count** | `17` symbols in `cognix.__all__` |
| **Supported Python Versions** | `3.10`, `3.11`, `3.12` |

> [!IMPORTANT]
> The PyPI distribution package name is **`cognix-uncertainty-framework`**, while the Python module import is **`cognix`**. Installing `pip install cognix` installs an unrelated legacy package or fails. Always specify `cognix-uncertainty-framework`.

---

## Canonical Installation Command

The canonical command to install the released v0.1.0 distribution is:

```bash
python -m pip install cognix-uncertainty-framework==0.1.0
```

### Why `python -m pip`?

Running `python -m pip` ensures that `pip` executes strictly against the specific Python interpreter selected by `python`, avoiding ambiguity when multiple Python interpreters or broken PATH environment variables exist on your system.

---

## Python Version Support & Verification

COGNIX officially targets and tests against:
- **Python 3.10**
- **Python 3.11**
- **Python 3.12**

These versions are verified in the GitHub Actions continuous integration pipeline (`tests.yml` and `release.yml`) across Linux and Windows environments.

> [!WARNING]
> While `pyproject.toml` specifies `requires-python = ">=3.10"`, newer Python interpreters (such as Python 3.13 or Python 3.14 preview builds) may encounter compatibility issues with upstream C-extension dependencies (such as `numpy` or `scipy`). Do not assume unverified Python versions are supported in production.

---

## Windows Installation & Interpreter Selection

Windows systems frequently have multiple Python interpreters installed simultaneously (e.g., Python 3.12 and Python 3.14). The Windows Python Launcher (`py.exe`) provides exact version targeting.

### 1. Inspect Available Interpreters
```powershell
py -0p
```
Example output:
```text
 -V:3.14 *        C:\Users\Username\AppData\Local\Programs\Python\Python314\python.exe
 -V:3.12          C:\Users\Username\AppData\Local\Programs\Python\Python312\python.exe
```

### 2. Create a Virtual Environment with Python 3.12
```powershell
# Create an isolated virtual environment using Python 3.12
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If script execution is restricted in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*.

### 3. Install COGNIX in the Virtual Environment
```powershell
python -m pip install --upgrade pip
python -m pip install cognix-uncertainty-framework==0.1.0
```

---

## Linux & macOS Installation

### 1. Ensure Supported Python (3.10, 3.11, or 3.12)
```bash
python3.12 --version || python3.11 --version || python3.10 --version
```

### 2. Create and Activate Virtual Environment
```bash
# Using Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Package
```bash
python -m pip install --upgrade pip
python -m pip install cognix-uncertainty-framework==0.1.0
```

---

## Verifying the Installation

To verify that COGNIX is correctly installed in your active environment, run the following one-line verification command:

```bash
python -c "import cognix; print(cognix.__version__); print(cognix.__file__); print(len(cognix.__all__))"
```

### Expected Output

```text
0.1.0
<path-to-your-env>/site-packages/cognix/__init__.py
17
```

Verify that:
1. The printed version is exactly `0.1.0`.
2. The file path resolves to your environment's `site-packages/cognix/__init__.py` (not a local uninstalled source repository).
3. The count of public exports in `cognix.__all__` is exactly `17`.

---

## Optional Dependency Extras

The base `cognix-uncertainty-framework` package is intentionally lightweight and depends only on:
- `numpy>=1.24`
- `scipy>=1.10`
- `pydantic>=2.0`
- `pyyaml>=6.0`

It does **not** mandate PyTorch, GPU toolchains, or simulator runtimes. If your application requires relational graph neural networks, explainability attribution, or vision tooling, install the respective extras:

```bash
# Graph Attention Networks (PyTorch + NetworkX)
python -m pip install "cognix-uncertainty-framework[graph]==0.1.0"

# Explainability & Attribution (SHAP + LIME)
python -m pip install "cognix-uncertainty-framework[explainability]==0.1.0"

# Vision Utilities (PyTorch + TorchVision + Pillow)
python -m pip install "cognix-uncertainty-framework[vision]==0.1.0"

# Telemetry Dashboard (FastAPI + Uvicorn + WebSockets)
python -m pip install "cognix-uncertainty-framework[dashboard]==0.1.0"

# Multi-Agent Reinforcement Learning (PettingZoo)
python -m pip install "cognix-uncertainty-framework[marl]==0.1.0"

# All Generic Extras Combined
python -m pip install "cognix-uncertainty-framework[all]==0.1.0"
```

---

## Installing for Local Development

To clone and work on the COGNIX codebase directly:

```bash
git clone https://github.com/AdityaR4685/Cognix_.git
cd Cognix_

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install in editable mode with development dependencies
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To run the core test suite:

```bash
pytest tests/ -m "not research_assets"
```
