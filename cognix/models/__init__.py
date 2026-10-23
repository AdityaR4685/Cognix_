"""
Advanced Models Module for COGNIX (PyTorch).
"""
try:
    import torch  # noqa: F401
    from .bayesian import BayesLinear, ELBOLoss
    from .evidential import DirichletLoss, EvidentialNetwork
    __all__ = ["BayesLinear", "ELBOLoss", "DirichletLoss", "EvidentialNetwork"]
except ImportError:
    pass
