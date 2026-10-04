"""
Real-data CarlAnomaly feature extractors (Experiment 1).

SCOPE (architecture guardrail): every function in this module is CARLA- /
CarlAnomaly-specific: it encodes the empirically verified real schemas of the
Base archive and CARLA sensor semantics. Nothing here may move into generic
COGNIX modules; the real agents consume these extractors and expose only the
canonical agent contract upstream.

Verified real schemas (byte-range probe of carlanomaly-base-test.tar.gz):
    gnss.feather: altitude, latitude, longitude            (float64, no index)
    imu.feather:  acceleration_x/y/z, compass, longitude_x/y/z
                                                          (float64, no index)

Design note (deterministic compact features, NOT pretrained embeddings): the
frozen synthetic extractors in features.py map handcrafted 1-2 dim statistics
to hazard probabilities. The extractors here produce compact DETERMINISTIC
handcrafted feature vectors for one-class normality models (the learning
happens in normality.py / real_agents.py over these vectors fitted on real
data). They are NOT learned perception and NOT pretrained visual embeddings;
a frozen backbone (ResNet/MobileNet-style) embedding remains a later
Experiment-1 option, upgradeable without touching the agent contract.
features.py stays untouched.

No metric-depth or sampling-rate claims are made anywhere: depth is not used
by Experiment 1, and no frequency is inferred from row counts (acceleration
differences are DISCRETE per-sample statistics, not SI jerk, because the
sampling interval is unknown).
"""
from __future__ import annotations

from math import cos, degrees, radians

import numpy as np

# Verified real IMU acceleration columns. compass / longitude_* are orientation
# and angular-rate channels and NEVER enter acceleration features.
IMU_ACCEL_COLUMNS = ("acceleration_x", "acceleration_y", "acceleration_z")
IMU_ORIENTATION_COLUMNS = ("compass", "longitude_x", "longitude_y", "longitude_z")

# Verified real GNSS columns.
GNSS_COLUMNS = ("altitude", "latitude", "longitude")


def camera_embedding_features(rgb_batch: np.ndarray) -> np.ndarray:
    """
    Compact DETERMINISTIC handcrafted camera features (NOT learned perception,
    NOT a pretrained visual embedding): block-mean intensities (4x4 grid) +
    variance + brightness, from a raw uint8 RGB array of shape (H, W, 3).

    This is a deliberately simple, dependency-free compact representation. It
    is still a genuine-perception INPUT for Experiment 1 only in the sense
    that the one-class model downstream learns the normal manifold over this
    vector from real data (nothing is hand-mapped to a hazard probability
    here); the representation itself is handcrafted and deterministic.
    Upgrading to a frozen pretrained backbone embedding later changes only
    this function and the manifest feature-hash.

    Returns a 1-D float64 feature vector (18 dims).
    """
    img = np.asarray(rgb_batch)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"camera features expect (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    f = img.astype(np.float64) / 255.0
    h, w = f.shape[:2]
    blocks = []
    for i in range(4):
        for j in range(4):
            patch = f[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            blocks.append(patch.mean())
    blocks.append(f.var())     # global variance
    blocks.append(f.mean())    # global brightness
    return np.asarray(blocks, dtype=np.float64)


def segmentation_histogram_features(seg_r_classes: np.ndarray, n_classes: int = 29) -> np.ndarray:
    """
    Class-composition histogram of a semantic-segmentation R-channel class map
    (verified: per-pixel class is encoded in the R channel; R values observed
    in official example: 1..28). L2-normalized so the feature is a composition.

    Returns a 1-D float64 vector of length n_classes.

    (H, W, 3) color inputs are accepted and the R channel (index 0) is used,
    matching the verified CarlAnomaly encoding (per-pixel class in R); extra
    channels are ignored rather than raveled in as class ids.
    """
    arr = np.asarray(seg_r_classes)
    if arr.ndim == 3:
        arr = arr[..., 0]  # verified encoding: per-pixel class lives in R
    classes = arr.ravel().astype(np.int64)
    if classes.min() < 0 or classes.max() >= n_classes:
        raise ValueError(
            f"segmentation class ids out of range [0, {n_classes}): "
            f"min={classes.min()}, max={classes.max()}"
        )
    hist = np.bincount(classes, minlength=n_classes).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError("empty segmentation map")
    return hist / total


def gnss_window_features(position_rows: np.ndarray) -> np.ndarray:
    """
    Trajectory-consistency features over a window of GNSS rows (verified
    schema: altitude, latitude, longitude; positional alignment row i <-> tick i).

    UNIT AUDIT (fixes an earlier defect): latitude and longitude are ANGULAR
    coordinates (degrees in real GNSS data) and altitude is a linear quantity
    (meters); the previous implementation differenced the degree columns
    directly as if they were interchangeable Euclidean coordinates, which is
    dimensionally incoherent (1 degree of latitude is ~111 km). This version
uses the standard LOCAL SMALL-AREA approximation (documented, not a CARLA
convention invention): over a short window,

        north_m ~= R * dlat_rad           (dlat in radians)
        east_m  ~= R * cos(lat_ref) * dlon_rad

    with R the mean Earth radius and lat_ref the window's mean latitude.
    Altitude is kept in meters. Degrees <-> meters round-trips use the SAME
    constants (see local_meter_offsets_to_degrees), so corruption recipes and
    features cannot drift apart.

    Features (8-dim float64, all finite):
        [mean/std east step (m), mean/std north step (m),
         mean/max horizontal step magnitude (m), altitude range (m),
         straightness = path_length / displacement (dimensionless)]
    """
    pos = np.asarray(position_rows, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 2:
        raise ValueError("gnss_window_features expects (n, 3) with n >= 2")
    if pos.shape[1] < 3:
        raise ValueError("gnss_window_features expects altitude/latitude/longitude columns")
    if not np.all(np.isfinite(pos)):
        raise ValueError("gnss_window_features expects finite rows")
    alt = pos[:, 0]
    lat_deg = pos[:, 1]
    lon_deg = pos[:, 2]
    lat_ref = float(lat_deg.mean())
    # Local tangent-plane projection (radians -> meters, small-area approx).
    north = np.diff(lat_deg) * (np.pi / 180.0) * EARTH_RADIUS_M
    east = np.diff(lon_deg) * (np.pi / 180.0) * EARTH_RADIUS_M * cos(radians(lat_ref))
    steps = np.stack([east, north], axis=1)
    step_norms = np.linalg.norm(steps, axis=1)
    path_len = float(step_norms.sum())
    disp_vec = np.array(
        [
            (lon_deg[-1] - lon_deg[0]) * (np.pi / 180.0) * EARTH_RADIUS_M * cos(radians(lat_ref)),
            (lat_deg[-1] - lat_deg[0]) * (np.pi / 180.0) * EARTH_RADIUS_M,
        ]
    )
    disp = float(np.linalg.norm(disp_vec))
    straightness = (path_len / disp) if disp > 1e-9 else 1.0
    return np.array(
        [
            east.mean(), east.std() if len(east) > 1 else 0.0,
            north.mean(), north.std() if len(north) > 1 else 0.0,
            step_norms.mean(), step_norms.max(),
            float(alt.max() - alt.min()),
            straightness,
        ],
        dtype=np.float64,
    )


# Mean Earth radius (IUGG mean R1 = (2a+b)/3), used ONLY for the documented
# local small-area degree<->meter approximation above. Not a CARLA convention.
EARTH_RADIUS_M = 6371008.8


def local_meter_offsets_to_degrees(north_m: float, east_m: float, lat_deg: float):
    """Convert small local offsets (meters) to (dlat_deg, dlon_deg) using the
    SAME spherical small-area approximation as gnss_window_features, so a
    corruption recipe expressed in meters lands in the feature space exactly
    as documented. Only valid for small offsets around `lat_deg`."""
    dlat = degrees(float(north_m) / EARTH_RADIUS_M)
    dlon = degrees(float(east_m) / (EARTH_RADIUS_M * cos(radians(float(lat_deg)))))
    return dlat, dlon


def imu_window_features(accel_rows: np.ndarray) -> np.ndarray:
    """
    Motion-signal features over a window of IMU acceleration rows (verified
    schema; uses ONLY acceleration_x/y/z — compass / longitude_* are excluded
    by construction, mirroring the corrected loader statistic).

    Features: per-axis mean/std, acceleration-norm mean/std, mean and max of
    the DISCRETE ACCELERATION-CHANGE statistic |diff(a_norm)|. The sampling
    interval is unknown and no rate is inferred from row counts, so this is a
    dimensionless per-sample change statistic, NOT physical jerk in SI units
    (m/s^3).
    """
    acc = np.asarray(accel_rows, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[0] < 2 or acc.shape[1] != 3:
        raise ValueError("imu_window_features expects (n, 3) with n >= 2")
    if not np.all(np.isfinite(acc)):
        raise ValueError("imu_window_features expects finite rows")
    a_norm = np.linalg.norm(acc, axis=1)
    delta = np.abs(np.diff(a_norm))
    feats = np.array(
        [
            acc[:, 0].mean(), acc[:, 0].std(),
            acc[:, 1].mean(), acc[:, 1].std(),
            acc[:, 2].mean(), acc[:, 2].std(),
            a_norm.mean(), a_norm.std(),
            float(delta.mean()), float(delta.max()),
        ],
        dtype=np.float64,
    )
    return feats
