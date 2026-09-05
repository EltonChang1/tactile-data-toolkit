"""Vision-tactile (optical gel) calibration pipeline."""

from tactile_toolkit.calibration.gelsight.calibrator import GelSightCalibrator, GelSightConfig
from tactile_toolkit.calibration.gelsight.forces import (
    ElastomerModel,
    dense_force_map,
    force_field,
    pool_mean,
)
from tactile_toolkit.calibration.gelsight.markers import MarkerTracker, detect_markers
from tactile_toolkit.calibration.gelsight.poisson import (
    depth_from_frames,
    poisson_cumsum,
    poisson_dst,
    poisson_reconstruct,
    rgb_to_gradients,
)
from tactile_toolkit.calibration.gelsight.reference import (
    ReferenceEstimator,
    estimate_reference,
    marker_mask,
)

__all__ = [
    "ElastomerModel",
    "GelSightCalibrator",
    "GelSightConfig",
    "MarkerTracker",
    "ReferenceEstimator",
    "dense_force_map",
    "depth_from_frames",
    "detect_markers",
    "estimate_reference",
    "force_field",
    "marker_mask",
    "poisson_cumsum",
    "poisson_dst",
    "poisson_reconstruct",
    "pool_mean",
    "rgb_to_gradients",
]
