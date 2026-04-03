"""Multi-camera video recording for deterministic debugging.

Records raw frames + tracking data from all cameras simultaneously.
Thread-safe — multiple camera tracker threads call record_frame() concurrently.
Each camera gets its own subdirectory with JPEG frames and a JSONL tracking log.

Recording starts only when all expected cameras are producing frames (or after
a configurable timeout for broken cameras).

Usage:
    session = RecordingSession(
        base_path="./recordings",
        cameras=["camera_head", "camera_left", "camera_right"],
        config_snapshot={...})
    session.record_frame("camera_head", frame, time.time())
    session.record_frame("camera_head", None, time.time(), tracking_data={...})
    session.stop()
"""

# builtin
import json
import os
import time
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Any

# 3rd party
import cv2
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class RecordingSession:
    """Records raw frames and tracking data from multiple cameras.

    Thread-safe: each camera tracker thread can call record_frame()
    concurrently. Per-camera frame counters and file handles are
    protected by locks.

    Args:
        base_path: Root directory for recordings.
        cameras: List of camera names to record.
        config_snapshot: Dict of config values to save in metadata.
        jpeg_quality: JPEG compression quality (0-100).
    """

    def __init__(self, base_path: str, cameras: List[str],
                 config_snapshot: Dict[str, Any] = None,
                 jpeg_quality: int = 85) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._jpeg_quality = jpeg_quality
        self._start_time = time.time()

        # Create session directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(base_path, f"session_{ts}")
        os.makedirs(self._session_dir, exist_ok=True)

        # Per-camera state
        self._cameras = cameras
        self._frame_counts: Dict[str, int] = {}
        self._locks: Dict[str, Lock] = {}
        self._jsonl_files: Dict[str, Any] = {}

        for cam in cameras:
            cam_dir = os.path.join(self._session_dir, cam)
            frames_dir = os.path.join(cam_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            self._frame_counts[cam] = 0
            self._locks[cam] = Lock()
            jsonl_path = os.path.join(cam_dir, "tracking.jsonl")
            self._jsonl_files[cam] = open(jsonl_path, 'w')

        self._config_snapshot = config_snapshot or {}
        self._active = True
        self.logger.info(f"Recording session started: {self._session_dir}")

    def record_frame(self, camera: str, frame: Optional[np.ndarray],
                     wall_time: float, tracking_data: Dict[str, Any] = None) -> None:
        """Record a frame and/or tracking data for a camera.

        Call twice per frame: once with the raw frame (before YOLO),
        once with tracking_data (after YOLO). Both share the same wall_time.

        Args:
            camera: Camera name (e.g., "camera_head").
            frame: Raw BGR numpy array, or None if only appending tracking data.
            wall_time: Wall-clock timestamp for cross-camera sync.
            tracking_data: Detection/tracking results dict, or None if only saving frame.
        """
        if not self._active or camera not in self._locks:
            return

        with self._locks[camera]:
            frame_num = self._frame_counts[camera]

            # Save raw frame as JPEG
            if frame is not None:
                frame_path = os.path.join(
                    self._session_dir, camera, "frames", f"{frame_num:06d}.jpg")
                cv2.imwrite(frame_path, frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
                self._frame_counts[camera] = frame_num + 1

            # Append tracking data to JSONL
            if tracking_data is not None:
                record = {
                    "frame_number": frame_num,
                    "wall_time": wall_time,
                    "tracking": tracking_data,
                }
                self._jsonl_files[camera].write(json.dumps(record) + "\n")
                self._jsonl_files[camera].flush()

    def stop(self) -> None:
        """Stop recording and write metadata."""
        if not self._active:
            return
        self._active = False
        duration = time.time() - self._start_time

        # Close JSONL files
        for cam, f in self._jsonl_files.items():
            f.close()

        # Write metadata
        metadata = {
            "session_id": os.path.basename(self._session_dir),
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "duration_s": round(duration, 1),
            "cameras": {},
            "config_snapshot": self._config_snapshot,
        }
        for cam in self._cameras:
            metadata["cameras"][cam] = {
                "total_frames": self._frame_counts.get(cam, 0),
            }

        meta_path = os.path.join(self._session_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(
            f"Recording stopped: {self._session_dir} "
            f"({duration:.1f}s, "
            f"{sum(self._frame_counts.values())} total frames)")

    @property
    def session_dir(self) -> str:
        """Return the session directory path."""
        return self._session_dir

    @property
    def is_active(self) -> bool:
        """Return True if recording is active."""
        return self._active


class RecordingGate:
    """Tracks which cameras are ready and starts recording when all are online.

    Handles broken cameras by starting after a timeout with whatever
    cameras are available.

    Args:
        cameras: List of expected camera names.
        timeout: Seconds to wait for all cameras before starting with available ones.
    """

    def __init__(self, cameras: List[str], timeout: float = 120.0) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._expected = set(cameras)
        self._ready: Dict[str, bool] = {cam: False for cam in cameras}
        self._start_time = time.time()
        self._timeout = timeout
        self._triggered = False

    def camera_ready(self, camera: str) -> None:
        """Mark a camera as producing frames."""
        if camera in self._ready and not self._ready[camera]:
            self._ready[camera] = True
            self.logger.info(f"Camera {camera} ready for recording")

    def should_start(self) -> bool:
        """Check if recording should begin.

        Returns True once — either when all cameras are ready or
        when the timeout expires with at least one camera available.
        """
        if self._triggered:
            return False

        all_ready = all(self._ready.values())
        timed_out = (time.time() - self._start_time) > self._timeout
        any_ready = any(self._ready.values())

        if all_ready:
            self._triggered = True
            self.logger.info("All cameras ready — starting recording")
            return True
        elif timed_out and any_ready:
            self._triggered = True
            ready_cams = [c for c, r in self._ready.items() if r]
            missing_cams = [c for c, r in self._ready.items() if not r]
            self.logger.warning(
                f"Recording timeout — starting with {ready_cams}, "
                f"missing: {missing_cams}")
            return True

        return False

    def get_ready_cameras(self) -> List[str]:
        """Return list of cameras that are producing frames."""
        return [cam for cam, ready in self._ready.items() if ready]
