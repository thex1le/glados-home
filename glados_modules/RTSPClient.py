# native imports
import os
from typing import Optional, Dict
from time import sleep, time
from threading import Thread, Lock

# 3rd party imports
import cv2

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import CameraEnum, LoggingEnums


class RtspConsumerError(Exception):
    pass


class RtspConsumer:
    def __init__(self, uri: str, location: str, reconnect_delay: int = 5) -> None:
        """Initializes the RtspConsumer.

        Runs a background thread that continuously reads frames and keeps
        only the latest. get_frame() always returns the freshest frame,
        never a stale buffered one.

        Args:
            uri: The RTSP URI of the stream.
            location: The location identifier for the consumer.
            reconnect_delay: Seconds to wait before attempting to reconnect.
        """
        self.rtsp_uri = uri
        self.location = location
        self.__name__ = f"{self.location}_rtsp_consumer"
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.reconnect_delay = reconnect_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_lock = Lock()
        self._latest_frame = None
        self._latest_resolution = (0, 0)
        self._running = True
        self.connect()
        self._reader = Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def connect(self) -> None:
        """
        Attempts to connect to the RTSP stream. Retries on failure.
        """
        # GStreamer pipelines for RTSP decode — tried in order until one connects
        # Pipeline 1: Original working pipeline (nvh264dec, no cudadownload)
        gst_original = (
            f"rtspsrc location={self.rtsp_uri} latency=0 ! "
            f"rtpjitterbuffer drop-on-latency=true ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"nvh264dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false emit-signals=false")
        # Pipeline 2: GPU decode with explicit CUDA download
        gst_hw_cuda = (
            f"rtspsrc location={self.rtsp_uri} latency=0 ! "
            f"rtpjitterbuffer drop-on-latency=true ! "
            f"rtph264depay ! "
            f"h264parse ! "
            f"nvh264dec ! "
            f"cudadownload ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false emit-signals=false")
        # Pipeline 3: Software decode fallback (low-latency to match hw pipelines)
        gst_sw = (
            f"rtspsrc location={self.rtsp_uri} latency=0 ! "
            f"rtpjitterbuffer drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false")
        attempt = 0
        while True:
            try:
                # Release any previous capture before creating a new pipeline
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    sleep(0.5)  # let GStreamer tear down before creating a new pipeline

                attempt += 1
                # Try each pipeline until one connects
                pipelines = [
                    (gst_original, "nvh264dec"),
                    (gst_hw_cuda, "nvh264dec+cudadownload"),
                    (gst_sw, "avdec_h264"),
                ]
                for pipeline, label in pipelines:
                    self.logger.info(f"Connecting to RTSP at {self.rtsp_uri} "
                                     f"(attempt {attempt}, {label})...")
                    self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if self.cap.isOpened():
                        self.logger.info(f"Connected to {self.rtsp_uri} using {label}.")
                        return
                    self.cap.release()
                    self.cap = None
                    sleep(0.5)

                self.logger.warning(
                    f"GStreamer pipelines failed for {self.rtsp_uri}. "
                    f"Retrying in {self.reconnect_delay}s...")
            except Exception as e:
                self.logger.error(f"Exception occurred while connecting: {e}")

            sleep(self.reconnect_delay)

    def _read_loop(self) -> None:
        """Background thread: continuously reads frames, keeps only the latest.

        This prevents GStreamer decode buffers from building up when the
        YOLO tracker thread can't consume frames as fast as they arrive.
        """
        while self._running:
            try:
                if not self.cap or not self.cap.isOpened():
                    sleep(0.5)
                    self.connect()
                    continue
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._latest_resolution = (
                            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        )
                else:
                    self.logger.warning(f"Read failed, reconnecting {self.rtsp_uri}...")
                    self.cap.release()
                    self.cap = None
                    sleep(self.reconnect_delay)
            except Exception as e:
                self.logger.error(f"Read loop error: {e}")
                sleep(1.0)

    def get_frame(self) -> Dict[str, Optional[any]]:
        """Returns the latest frame from the background reader.

        Blocks until the first frame is available, then always returns
        immediately with the freshest frame (never a stale buffered one).
        """
        while self._running:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    resolution = self._latest_resolution
                    break
            sleep(0.01)
        return {
            CameraEnum.MSG_LOCATION_KEY.value: self.location,
            CameraEnum.MSG_RAW_IMAGE.value: frame,
            CameraEnum.MSG_RESOLUTION.value: resolution,
        }

    def close(self) -> None:
        """
        Closes the VideoCapture resource and releases all connections.
        """
        self._running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.logger.info(f"{self.__name__} Resources released.")
        else:
            self.logger.info(f"{self.__name__} Resources were already released or never opened.")


class VideoFileSource:
    """Replay recorded video frames as if from a live camera.

    Returns the same image_dict structure as RtspConsumer.get_frame().
    Can run at original wall-clock timing or as fast as possible.

    Args:
        session_path: Path to the recording session directory.
        camera: Camera name subdirectory (e.g., "camera_head").
        realtime: If True, sleep between frames to match original timing.
    """

    def __init__(self, session_path: str, camera: str, realtime: bool = True) -> None:
        import json
        from glob import glob

        self.__name__ = f"{camera}_video_source"
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._camera = camera
        self._realtime = realtime

        # Load frame list
        frames_dir = os.path.join(session_path, camera, "frames")
        self._frame_list = sorted(glob(os.path.join(frames_dir, "*.jpg")))
        self._index = 0

        # Load metadata for resolution
        meta_path = os.path.join(session_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}

        # Load tracking JSONL for wall_time sync
        self._wall_times: list = []
        tracking_path = os.path.join(session_path, camera, "tracking.jsonl")
        if os.path.exists(tracking_path):
            with open(tracking_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        self._wall_times.append(record.get("wall_time", 0))

        self.logger.info(f"Video source loaded: {len(self._frame_list)} frames from {frames_dir}")

    def get_frame(self) -> Dict[str, Optional[any]]:
        """Return the next frame as an image_dict matching RtspConsumer format.

        Returns:
            Dict with camera location, raw image, and resolution.

        Raises:
            StopIteration: When all frames have been replayed.
        """
        if self._index >= len(self._frame_list):
            raise StopIteration(f"Recording complete: {len(self._frame_list)} frames replayed")

        frame = cv2.imread(self._frame_list[self._index])
        if frame is None:
            self.logger.warning(f"Failed to read frame: {self._frame_list[self._index]}")
            self._index += 1
            return self.get_frame()

        # Realtime pacing — sleep to match original frame timing
        if self._realtime and self._index > 0 and self._index < len(self._wall_times):
            dt = self._wall_times[self._index] - self._wall_times[self._index - 1]
            if dt > 0:
                sleep(dt)

        h, w = frame.shape[:2]
        self._index += 1

        return {
            CameraEnum.MSG_LOCATION_KEY.value: self._camera,
            CameraEnum.MSG_RAW_IMAGE.value: frame,
            CameraEnum.MSG_RESOLUTION.value: (float(w), float(h)),
        }

    def close(self) -> None:
        """No-op for file source — nothing to release."""
        pass


if __name__ == "__main__":
    # Standalone debugging: connect to an RTSP stream and display a frame
    import argparse
    import sys

    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="RTSP Consumer Test Handler")
    # Define command-line arguments
    parser.add_argument('--uri', type=str, required=True,
                        help='RTSP stream URI (e.g., rtsp://192.168.1.100:8554/test)')
    parser.add_argument('--location', type=str, required=True,
                        help='Location identifier for the RTSP stream (e.g., FrontDoor)')
    # Parse the arguments
    args = parser.parse_args()
    # Initialize RtspConsumer
    try:
        consumer = RtspConsumer(uri=args.uri, location=args.location)
        consumer.logger.info("RtspConsumer initialized successfully.")
    except RtspConsumerError as e:
        print(f"Failed to initialize RtspConsumer: {e}")
        sys.exit(1)
    # Retrieve a frame
    frame_data = consumer.get_frame()
    # Extract information from the frame data
    location = frame_data.get(CameraEnum.MSG_LOCATION_KEY.value, "Unknown")
    frame = frame_data.get(CameraEnum.MSG_RAW_IMAGE.value, None)
    resolution = frame_data.get(CameraEnum.MSG_RESOLUTION.value, (0, 0))
    # Display the retrieved information
    print(f"Location: {location}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(frame)
