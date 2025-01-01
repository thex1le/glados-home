# 3rd party imports
import cv2
from time import sleep

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import CameraEnum


class RtspConsumerError(Exception):
    pass


class RtspConsumer:
    def __init__(self, uri: str, location: str) -> None:
        self.rtsp_uri = uri
        self.location = location
        self.__name__ = f"{self.location}_rtsp_consumer"
        self.logger = setup_logger(name=self.__name__)
        # GStreamer pipeline that sets max-lateness to ensure only the latest frame is captured
        gst_pipeline = (
            f"rtspsrc location={self.rtsp_uri} latency=0 ! "
            f"rtpjitterbuffer drop-on-latency=true ! "
            f"decodebin ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false emit-signals=false"
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    def get_frame(self) -> dict:
        """
        Return a from a rtsp stream frame for further processing
        """
        if not self.cap.isOpened():
            msg = f"Error: Cannot open RTSP stream at {self.rtsp_uri}"
            self.logger.error(msg)
            raise RtspConsumerError(msg)
        image_dict = {CameraEnum.MSG_LOCATION_KEY.value: self.location,
                      CameraEnum.MSG_RAW_IMAGE.value: None,
                      CameraEnum.MSG_RESOLUTION.value: (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                        self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                                        )}
        frame = None
        c = 0
        while frame is None:
            ret, frame = self.cap.read()
            if not ret:
                if c >= 10:
                    self.logger.debug(f"{self.__name__} failed to retrieve frame {c} times exiting")
                    break
                c += 1
        image_dict[CameraEnum.MSG_RAW_IMAGE.value] = frame
        return image_dict

    def close(self) -> None:
        """
        Close out all connections
        """
        self.cap.release()
        self.logger.info(f"{self.__name__} Resources released")


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
