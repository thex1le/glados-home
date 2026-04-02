# native imports
from typing import Optional, Dict
from time import sleep

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
        self.connect()

    def connect(self) -> None:
        """
        Attempts to connect to the RTSP stream. Retries on failure.
        """
        gst_pipeline = (
            f"rtspsrc location={self.rtsp_uri} latency=0 ! "
            f"rtpjitterbuffer drop-on-latency=true ! "
            f"rtph264depay ! "
            f"h264parse ! "  # Parse the stream to negotiate caps properly
            f"nvh264dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false emit-signals=false"
            )
        attempt = 0
        while True:
            try:
                attempt += 1
                self.logger.info(f"Connecting to RTSP stream at {self.rtsp_uri} (attempt {attempt})...")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    self.logger.info(f"Successfully connected to RTSP stream {self.rtsp_uri}.")
                    return
                else:
                    self.logger.warning(
                        f"Failed to connect to RTSP stream. Retrying in {self.reconnect_delay}s...")
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                self.logger.error(f"Exception occurred while connecting: {e}")

            sleep(self.reconnect_delay)

    def get_frame(self) -> Dict[str, Optional[any]]:
        """
        Retrieves a frame from the RTSP stream. Blocks until a frame is successfully retrieved.

        :return: A dictionary containing the location, raw image, and resolution.
        :raises RtspConsumerError: If unable to retrieve a frame after multiple attempts.
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.warning("VideoCapture not opened. Attempting to reconnect...")
            self.connect()

        image_dict = {
            CameraEnum.MSG_LOCATION_KEY.value: self.location,
            CameraEnum.MSG_RAW_IMAGE.value: None,
            CameraEnum.MSG_RESOLUTION.value: (
                self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
        }

        while True:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                image_dict[CameraEnum.MSG_RAW_IMAGE.value] = frame
                return image_dict
            else:
                self.logger.warning(f"Failed to retrieve frame. Reconnecting...")
                self.cap.release()
                self.cap = None
                sleep(self.reconnect_delay)
                self.connect()

    def close(self) -> None:
        """
        Closes the VideoCapture resource and releases all connections.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.logger.info(f"{self.__name__} Resources released.")
        else:
            self.logger.info(f"{self.__name__} Resources were already released or never opened.")


if __name__ == "__main__":
    # stand allow debugging stub
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
