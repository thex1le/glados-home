"""Tests for video recording and replay: RecordingSession, RecordingGate, VideoFileSource."""

import json
import os
import time
import pytest
import numpy as np

from glados_modules.VideoRecorder import RecordingSession, RecordingGate


class TestRecordingSession:
    """Test multi-camera frame recording."""

    def test_creates_session_directory(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head"])
        assert os.path.isdir(session.session_dir)
        session.stop()

    def test_creates_per_camera_directories(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head", "camera_left"])
        assert os.path.isdir(os.path.join(session.session_dir, "camera_head", "frames"))
        assert os.path.isdir(os.path.join(session.session_dir, "camera_left", "frames"))
        session.stop()

    def test_records_frame_as_jpeg(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head"])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.record_frame("camera_head", frame, time.time())
        frames_dir = os.path.join(session.session_dir, "camera_head", "frames")
        assert os.path.exists(os.path.join(frames_dir, "000000.jpg"))
        session.stop()

    def test_records_tracking_data_to_jsonl(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head"])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        wt = time.time()
        session.record_frame("camera_head", frame, wt)
        session.record_frame("camera_head", None, wt,
                             tracking_data={"person": {"count": 1}})
        session.stop()
        jsonl_path = os.path.join(session.session_dir, "camera_head", "tracking.jsonl")
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["tracking"]["person"]["count"] == 1

    def test_frame_counter_increments(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head"])
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            session.record_frame("camera_head", frame, time.time())
        frames_dir = os.path.join(session.session_dir, "camera_head", "frames")
        assert os.path.exists(os.path.join(frames_dir, "000002.jpg"))
        session.stop()

    def test_writes_metadata_on_stop(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head", "camera_left"],
                                    config_snapshot={"alpha": 0.7})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.record_frame("camera_head", frame, time.time())
        session.stop()
        meta_path = os.path.join(session.session_dir, "metadata.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        assert meta["cameras"]["camera_head"]["total_frames"] == 1
        assert meta["cameras"]["camera_left"]["total_frames"] == 0
        assert meta["config_snapshot"]["alpha"] == 0.7

    def test_thread_safe_multi_camera(self, tmp_path):
        """Multiple cameras recording concurrently."""
        from threading import Thread
        session = RecordingSession(str(tmp_path), ["cam_a", "cam_b"])

        def record_n(cam, n):
            for _ in range(n):
                frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                session.record_frame(cam, frame, time.time())

        t1 = Thread(target=record_n, args=("cam_a", 10))
        t2 = Thread(target=record_n, args=("cam_b", 15))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        session.stop()

        meta_path = os.path.join(session.session_dir, "metadata.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        assert meta["cameras"]["cam_a"]["total_frames"] == 10
        assert meta["cameras"]["cam_b"]["total_frames"] == 15

    def test_ignores_unknown_camera(self, tmp_path):
        session = RecordingSession(str(tmp_path), ["camera_head"])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not crash
        session.record_frame("camera_unknown", frame, time.time())
        session.stop()


class TestRecordingGate:
    """Test camera readiness gating."""

    def test_starts_when_all_ready(self):
        gate = RecordingGate(["cam_a", "cam_b"])
        gate.camera_ready("cam_a")
        assert not gate.should_start()
        gate.camera_ready("cam_b")
        assert gate.should_start()

    def test_only_triggers_once(self):
        gate = RecordingGate(["cam_a"])
        gate.camera_ready("cam_a")
        assert gate.should_start()
        assert not gate.should_start()  # second call returns False

    def test_timeout_starts_with_available(self):
        gate = RecordingGate(["cam_a", "cam_b"], timeout=0.1)
        gate.camera_ready("cam_a")
        time.sleep(0.2)
        assert gate.should_start()  # cam_b never came online

    def test_no_cameras_no_start(self):
        gate = RecordingGate(["cam_a"], timeout=0.1)
        time.sleep(0.2)
        assert not gate.should_start()  # no cameras ready at all

    def test_get_ready_cameras(self):
        gate = RecordingGate(["cam_a", "cam_b", "cam_c"])
        gate.camera_ready("cam_a")
        gate.camera_ready("cam_c")
        ready = gate.get_ready_cameras()
        assert "cam_a" in ready
        assert "cam_c" in ready
        assert "cam_b" not in ready


class TestVideoFileSource:
    """Test replay frame source from recorded JPEG files."""

    def _create_session(self, tmp_path, camera="camera_head", n_frames=5):
        """Create a fake recording session for testing."""
        import cv2
        cam_dir = os.path.join(str(tmp_path), camera)
        frames_dir = os.path.join(cam_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Write frames
        for i in range(n_frames):
            frame = np.full((480, 640, 3), i * 50, dtype=np.uint8)
            cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.jpg"), frame)

        # Write tracking JSONL with wall_times
        base_time = time.time()
        with open(os.path.join(cam_dir, "tracking.jsonl"), 'w') as f:
            for i in range(n_frames):
                record = {"frame_number": i, "wall_time": base_time + i * 0.1}
                f.write(json.dumps(record) + "\n")

        # Write metadata
        with open(os.path.join(str(tmp_path), "metadata.json"), 'w') as f:
            json.dump({"cameras": {camera: {"resolution": [640, 480]}}}, f)

        return str(tmp_path)

    def test_loads_frames(self, tmp_path):
        from glados_modules.RTSPClient import VideoFileSource
        session_path = self._create_session(tmp_path)
        source = VideoFileSource(session_path, "camera_head", realtime=False)
        frame_dict = source.get_frame()
        assert frame_dict["camera"] == "camera_head"
        assert frame_dict["raw"] is not None
        assert frame_dict["raw"].shape == (480, 640, 3)

    def test_returns_all_frames(self, tmp_path):
        from glados_modules.RTSPClient import VideoFileSource
        session_path = self._create_session(tmp_path, n_frames=3)
        source = VideoFileSource(session_path, "camera_head", realtime=False)
        frames = []
        for _ in range(3):
            frames.append(source.get_frame())
        assert len(frames) == 3

    def test_raises_stop_iteration(self, tmp_path):
        from glados_modules.RTSPClient import VideoFileSource
        session_path = self._create_session(tmp_path, n_frames=1)
        source = VideoFileSource(session_path, "camera_head", realtime=False)
        source.get_frame()
        with pytest.raises(StopIteration):
            source.get_frame()

    def test_resolution_from_metadata(self, tmp_path):
        from glados_modules.RTSPClient import VideoFileSource
        session_path = self._create_session(tmp_path)
        source = VideoFileSource(session_path, "camera_head", realtime=False)
        frame_dict = source.get_frame()
        assert frame_dict["resolution"] == (640.0, 480.0)
