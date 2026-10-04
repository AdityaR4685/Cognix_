import numpy as np
from cognix.adapters.carla.normality import BootstrapNormalityEnsemble, EnsemblePredictiveCalibrator, MahalanobisNormality
from cognix.adapters.carla.real_agents import RealCameraAgent, RealIMUAgent, RealSegAgent

NODE_ORDER = ("Camera", "IMU", "Seg")

ADJACENCY = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)

OFFSETS = {"camera": (0, 18), "seg": (18, 47), "gnss": (47, 55), "imu": (55, 65)}

class GraphExportError(RuntimeError):
    """A frozen dependency or scientific contract failed; never substitute data."""

def require(condition, message):
    if not condition:
        raise GraphExportError(message)

def restore_agents(parameters, handoff):
    """Restore unchanged production classes from saved state, without any fit call."""
    result = {}
    for name, cls in zip(NODE_ORDER, (RealCameraAgent, RealIMUAgent, RealSegAgent)):
        mapping = handoff["calibration_parameters_raw"][name]
        audit = handoff["calibration_audit"][name]
        require(mapping.get("scientific_valid") is True and audit.get("finite_optimum") is True,
                f"{name}: invalid frozen calibration")
        require(all(isinstance(mapping.get(k), (int, float)) and np.isfinite(mapping[k])
                    for k in ("slope", "intercept")) and mapping["slope"] > 0,
                f"{name}: invalid/null mapping parameters")
        agent = cls(seed=42, n_members=5)
        ensemble = BootstrapNormalityEnsemble(n_members=5, seed=42)
        lo, hi = OFFSETS[name.lower()]
        for k in range(5):
            prefix = f"{name.lower()}_{k}_"
            member = MahalanobisNormality()
            member._mean = np.array(parameters[prefix + "mean"], dtype=np.float64, copy=True)
            member._prec = np.array(parameters[prefix + "precision"], dtype=np.float64, copy=True)
            member._d2_scale = float(parameters[prefix + "distance_scale"])
            require(member._mean.shape == (hi - lo,) and member._prec.shape == (hi - lo, hi - lo)
                    and np.isfinite(member._mean).all() and np.isfinite(member._prec).all()
                    and np.isfinite(member._d2_scale) and member._d2_scale > 0,
                    f"{name}: invalid member state")
            ensemble._members.append(member)
        calibrator = EnsemblePredictiveCalibrator()
        calibrator.a, calibrator.b = mapping["slope"], mapping["intercept"]
        calibrator._fitted = True
        calibrator._validity = dict(audit)
        calibrator._nll_initial = audit["nll_initial"]
        calibrator._nll_final = audit["nll_final"]
        calibrator._n_iter = audit["n_iter"]
        calibrator._converged = audit["converged"]
        agent._ensemble, agent._calibrator = ensemble, calibrator
        result[name] = agent
    return result
