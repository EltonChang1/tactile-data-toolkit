"""MP4 encoding with PyAV (bundles FFmpeg, no system install required)."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PREFERRED_CODECS = ("libx264", "libopenh264", "h264", "libsvtav1", "libaom-av1", "mpeg4")


def available_codec(preferred: str | None = None) -> str:
    """Return the first usable encoder name from ``preferred`` then :data:`PREFERRED_CODECS`."""
    import av

    candidates = [preferred] if preferred else []
    candidates += [c for c in PREFERRED_CODECS if c != preferred]
    for name in candidates:
        if not name:
            continue
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:  # noqa: BLE001 - codec missing in this build
            continue
    raise RuntimeError("No usable video encoder found in the PyAV build")


class VideoEncoder:
    """Stream RGB uint8 frames into an MP4 file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        codec: str | None = None,
        pix_fmt: str = "yuv420p",
        crf: int = 23,
        keyframe_interval: int = 2,
    ):
        import av

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = available_codec(codec)
        self.pix_fmt = pix_fmt
        self.crf = crf
        self.keyframe_interval = keyframe_interval
        self._av = av
        self._container: Any = None
        self._stream: Any = None
        self.frames = 0
        self.width = 0
        self.height = 0

    def _open(self, height: int, width: int) -> None:
        av = self._av
        # Most codecs and yuv420p require even dimensions.
        self.height = height - (height % 2)
        self.width = width - (width % 2)
        if (self.height, self.width) != (height, width):
            log.warning(
                "Cropping video frames from %dx%d to even size %dx%d",
                width,
                height,
                self.width,
                self.height,
            )
        self._container = av.open(str(self.path), mode="w")
        rate = Fraction(self.fps).limit_denominator(1000)
        self._stream = self._container.add_stream(self.codec, rate=rate)
        self._stream.width = self.width
        self._stream.height = self.height
        self._stream.pix_fmt = self.pix_fmt
        opts: dict[str, str] = {"g": str(self.keyframe_interval)}
        if self.codec in ("libx264", "libx265", "libsvtav1", "libaom-av1"):
            opts["crf"] = str(self.crf)
        if self.codec == "libx264":
            opts["preset"] = "veryfast"
        self._stream.options = opts
        # Let the encoder use frame- and slice-level threading.
        try:
            self._stream.thread_type = "AUTO"
        except Exception:  # noqa: BLE001 - not every codec exposes threading
            pass

    def write(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must be uint8 with shape (T, H, W, 3)")
        if self._container is None:
            self._open(frames.shape[1], frames.shape[2])
        av = self._av
        for frame in frames:
            crop = np.ascontiguousarray(frame[: self.height, : self.width])
            vf = av.VideoFrame.from_ndarray(crop, format="rgb24")
            vf.pts = self.frames
            for packet in self._stream.encode(vf):
                self._container.mux(packet)
            self.frames += 1

    def close(self) -> dict[str, Any]:
        if self._container is None:
            return {}
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None
        return probe_video(self.path)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return LeRobot-style ``video.*`` info for an encoded file."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        return {
            "video.height": int(stream.height),
            "video.width": int(stream.width),
            "video.codec": stream.codec_context.name,
            "video.pix_fmt": stream.codec_context.pix_fmt,
            "video.is_depth_map": False,
            "video.fps": float(rate) if rate else None,
            "video.channels": 3,
            "has_audio": len(container.streams.audio) > 0,
        }


def read_video_frames(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    """Decode an MP4 into ``(T, H, W, 3)`` uint8 RGB (for tests and inspection)."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)
