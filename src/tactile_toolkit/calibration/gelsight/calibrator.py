"""Chunk-wise vision-tactile calibration: frames -> depth, forces, contact, point cloud."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tactile_toolkit import schema as S
from tactile_toolkit.calibration.gelsight.forces import ElastomerModel, force_field
from tactile_toolkit.calibration.gelsight.markers import MarkerTracker
from tactile_toolkit.calibration.gelsight.poisson import DepthMethod, depth_from_frames
from tactile_toolkit.calibration.gelsight.reference import ReferenceEstimator, marker_mask
from tactile_toolkit.calibration.geometry import SensorGeometry
from tactile_toolkit.calibration.pointcloud import build_point_cloud
from tactile_toolkit.types import Modality, SensorMetadata, TactileChunk


@dataclass
class GelSightConfig:
    sensor_width_m: float = 0.020
    """Physical width of the imaged gel area; sets the pixel size."""
    pixel_size_m: float | None = None
    """Overrides ``sensor_width_m / width`` when given."""
    gradient_gain: float = 1.0
    depth_method: DepthMethod = "dst"
    reference_frames: int = 10
    mask_markers: bool = True
    track_markers: bool = True
    lk_window: int = 21
    lk_levels: int = 3
    stride: int = 8
    """Pixel block size per point-cloud element (``N = (H//stride) * (W//stride)``)."""
    store_depth_map: bool = True
    store_raw_image: bool = True
    workers: int = 0
    """Threads for depth reconstruction and tracking; ``0`` picks ``min(cpu_count, 8)``."""
    elastomer: ElastomerModel = field(default_factory=ElastomerModel)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["elastomer"] = self.elastomer.to_dict()
        return d


class GelSightCalibrator:
    """Stateful per-trajectory calibrator for optical gel sensors.

    Feed consecutive frame chunks to :meth:`process`. The first
    ``reference_frames`` frames define the no-contact reference; marker
    positions are detected on it once and tracked in every later frame.
    """

    def __init__(self, config: GelSightConfig | None = None, sensor_name: str = "gelsight"):
        self.config = config or GelSightConfig()
        self.sensor_name = sensor_name
        self._ref = ReferenceEstimator(self.config.reference_frames)
        self._tracker: MarkerTracker | None = None
        self._mask: np.ndarray | None = None
        self._geometry: SensorGeometry | None = None
        self._pixel_size: float | None = None
        self._image_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------- properties
    @property
    def modality(self) -> Modality:
        return Modality.VISION_TACTILE

    @property
    def geometry(self) -> SensorGeometry:
        if self._geometry is None:
            raise RuntimeError("Calibrator has not seen any frames yet")
        return self._geometry

    @property
    def reference(self) -> np.ndarray:
        return self._ref.reference

    @property
    def pixel_size_m(self) -> float:
        if self._pixel_size is None:
            raise RuntimeError("Calibrator has not seen any frames yet")
        return self._pixel_size

    def metadata(self) -> SensorMetadata:
        geo = self.geometry
        H, W = self._image_shape or (0, 0)
        return SensorMetadata(
            sensor_name=self.sensor_name,
            modality=self.modality,
            num_elements=geo.num_elements,
            layout=dict(geo.layout),
            units={
                "normal_force": "N",
                "shear_force": "N",
                "point_cloud": "m, N",
                "depth_map": "m",
            },
            extra={
                "image_height": H,
                "image_width": W,
                "config": self.config.to_dict(),
                "num_markers": self._tracker.num_markers if self._tracker else 0,
            },
        )

    # ---------------------------------------------------------------- setup
    def _ensure_setup(self, frames: np.ndarray) -> None:
        if self._image_shape is None:
            H, W = frames.shape[1:3]
            self._image_shape = (H, W)
            self._pixel_size = self.config.pixel_size_m or self.config.sensor_width_m / W
            self._geometry = SensorGeometry.from_image(H, W, self._pixel_size, self.config.stride)
        if not self._ref.ready:
            self._ref.feed(frames)
            if not self._ref.ready:
                self._ref.force_ready()
            ref = self._ref.reference
            if self.config.mask_markers or self.config.track_markers:
                self._mask = marker_mask(ref) if self.config.mask_markers else None
            if self.config.track_markers:
                self._tracker = MarkerTracker(
                    ref, win_size=self.config.lk_window, max_level=self.config.lk_levels
                )

    # -------------------------------------------------------------- process
    def process(self, frames: np.ndarray) -> TactileChunk:
        """Calibrate a ``(T, H, W, 3)`` uint8 chunk into schema arrays."""
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (T, H, W, 3)")
        self._ensure_setup(frames)
        cfg = self.config
        ref = self._ref.reference
        H, W = frames.shape[1:3]
        px = self.pixel_size_m

        depth = depth_from_frames(
            frames,
            ref,
            pixel_size_m=px,
            gain=cfg.gradient_gain,
            method=cfg.depth_method,
            marker_mask=self._mask if cfg.mask_markers else None,
            workers=cfg.workers,
        )

        shear_disp = None
        if self._tracker is not None and self._tracker.num_markers >= 3:
            disp, valid = self._tracker.track(frames, workers=cfg.workers)
            # Sample the shear field directly on the point-cloud grid.
            shear_disp = self._tracker.grid_field(disp, valid, (H, W), cfg.stride)

        normal, shear, indentation = force_field(depth, shear_disp, px, cfg.elastomer, cfg.stride)
        contact = cfg.elastomer.contact_mask(indentation)
        cloud = build_point_cloud(self.geometry, normal, shear, indentation)

        out = TactileChunk()
        out[S.CONTACT_MASK] = contact
        out[S.NORMAL_FORCE] = normal.astype(np.float32, copy=False)
        out[S.SHEAR_FORCE] = shear.astype(np.float32, copy=False)
        out[S.POINT_CLOUD] = cloud
        if cfg.store_raw_image:
            out[S.RAW_IMAGE] = frames.astype(np.uint8, copy=False)
        if cfg.store_depth_map:
            out[S.DEPTH_MAP] = depth.astype(np.float32, copy=False)
        return out
