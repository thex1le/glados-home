#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

from picamera2 import Picamera2

import sys
import threading
import time
from queue import Queue, Full, Empty

import numpy as np
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RtspFactory(GstRtspServer.RTSPMediaFactory):
    """
    Custom RTSP Media Factory that accepts I420 frames from a queue and streams them via RTSP.
    """
    def __init__(self, cam_x, cam_y, fps=30, frame_queue=None, **properties):
        super(RtspFactory, self).__init__(**properties)
        self.cam_x = int(cam_x)
        self.cam_y = int(cam_y)
        self.fps = fps
        self.frame_queue = frame_queue if frame_queue is not None else Queue(maxsize=30)  # Increased queue size
        self.appsrc = None  # Will be set in do_configure

        # Define the GStreamer pipeline with videoconvert to ensure I420 format
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
            "caps=video/x-raw,format=I420,width={},height={},framerate={}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            "! x264enc speed-preset=ultrafast tune=zerolatency "
            "! rtph264pay config-interval=1 name=pay0 pt=96"
        ).format(self.cam_x, self.cam_y, self.fps)
        logger.debug(f"RTSP Factory launch string: {self.launch_string}")

    def send_data(self, data):
        """
        Thread-safe method to enqueue the latest frame to be pushed to GStreamer.
        If the queue is full, drop the oldest frame to make room for the new one.
        """
        try:
            self.frame_queue.put_nowait(data)
            logger.debug("Frame enqueued successfully.")
        except Full:
            try:
                # Drop the oldest frame
                dropped_frame = self.frame_queue.get_nowait()
                logger.warning("Queue full. Dropped the oldest frame to enqueue the new one.")
                self.frame_queue.put_nowait(data)
                logger.debug("New frame enqueued after dropping the oldest frame.")
            except Empty:
                logger.error("Queue was full and empty upon attempting to drop a frame. Dropping the new frame.")
    
    def do_create_element(self, url):
        """
        Creates the GStreamer pipeline for the RTSP stream.
        """
        try:
            pipeline = Gst.parse_launch(self.launch_string)
            logger.info("GStreamer pipeline created successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to parse launch string: {e}")
            traceback.print_exc()
            return None

    def do_configure(self, rtsp_media):
        """
        Configures the appsrc element to handle incoming frames.
        """
        try:
            self.appsrc = rtsp_media.get_element().get_child_by_name('source')
            if self.appsrc:
                self.appsrc.set_property("emit-signals", True)
                self.appsrc.connect("need-data", self.on_need_data)
                logger.info("Configured appsrc for receiving data.")
            else:
                logger.error("appsrc element not found in pipeline.")
        except Exception as e:
            logger.error(f"Exception in do_configure: {e}")
            traceback.print_exc()

    def on_need_data(self, src, length):
        """
        Callback triggered when GStreamer needs more data.
        Pulls the next frame from the queue and pushes it into the pipeline.
        """
        try:
            frame = self.frame_queue.get(timeout=1 / self.fps)
            logger.debug("Frame dequeued for pushing.")
        except Empty:
            logger.warning("Frame queue is empty. Pushing silence (no frame).")
            return

        try:
            # Calculate expected buffer size for I420
            expected_size = self.cam_x * self.cam_y * 3 // 2  # I420 size
            actual_size = frame.size
            if actual_size != expected_size:
                logger.error(f"Invalid frame size: expected {expected_size}, got {actual_size}")
                return

            data_bytes = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(data_bytes)

            # Set timestamp and duration
            buf.duration = Gst.SECOND // self.fps
            # Calculate PTS based on current time
            buf.pts = buf.dts = int(time.time() * Gst.SECOND)

            retval = src.emit("push-buffer", buf)
            if retval == Gst.FlowReturn.OK:
                logger.debug("Frame pushed to GStreamer pipeline.")
            else:
                logger.error(f"Push buffer error: {retval}")

        except Exception as e:
            logger.error(f"Exception in on_need_data: {e}")
            traceback.print_exc()

class GstRtspServerThread(threading.Thread):
    """
    Thread to run the GStreamer RTSP server's main loop.
    """
    def __init__(self, factory, mount_point="/test", port=8554):
        super(GstRtspServerThread, self).__init__()
        self.factory = factory
        self.mount_point = mount_point
        self.port = port
        self.daemon = True  # Allows thread to exit when main program exits

    def run(self):
        try:
            # Create the RTSP server
            server = GstRtspServer.RTSPServer()
            server.set_service(str(self.port))
            logger.debug(f"RTSP Server set to port {self.port}.")

            # Get the mount points and add the factory
            mounts = server.get_mount_points()
            mounts.add_factory(self.mount_point, self.factory)
            logger.info(f"Mount point '{self.mount_point}' added.")

            # Attach the server to the default main context
            server.attach(None)
            logger.info(f"RTSP server started at rtsp://<PI_IP>:{self.port}{self.mount_point}")

            # Start the GLib main loop to handle RTSP requests
            loop = GLib.MainLoop()
            loop.run()
        except Exception as e:
            logger.error(f"RTSP server error: {e}")
            traceback.print_exc()

def main():
    # Initialize GStreamer
    Gst.init(None)
    logger.info("GStreamer initialized.")

    # Camera configuration parameters
    cam_res_x = 640      # Width of the video frame
    cam_res_y = 480      # Height of the video frame
    fps = 30             # Frames per second

    # Initialize PiCamera2
    picam2 = Picamera2()
    try:
        video_config = picam2.create_video_configuration(
            main={"size": (cam_res_x, cam_res_y), "format": "YUV420"},
            controls={"FrameRate": fps}
        )
        picam2.configure(video_config)
        picam2.start()
        logger.info("PiCamera2 started with YUV420 format.")
    except Exception as e:
        logger.error(f"Failed to start PiCamera2: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Initialize the RTSP factory for I420 with a frame queue
    frame_queue = Queue(maxsize=30)  # Increased queue size
    factory = RtspFactory(cam_x=cam_res_x, cam_y=cam_res_y, fps=fps, frame_queue=frame_queue)

    # Start the RTSP server in a separate thread
    rtsp_thread = GstRtspServerThread(factory, mount_point="/test", port=8554)
    rtsp_thread.start()
    logger.info("RTSP server thread started.")

    # Allow some time for the RTSP server to start
    time.sleep(1)

    logger.info("Starting frame capture and streaming.")

    # Main loop to capture frames and send to RTSP server
    try:
        while True:
            # Capture a frame from PiCamera2
            frame = picam2.capture_array("main")  # Returns a NumPy array in YUV420
            if frame is not None:
                # Enqueue the frame for RTSP streaming
                factory.send_data(frame)
                logger.debug(f"Captured frame shape: {frame.shape}, size: {frame.size}")
            else:
                logger.warning("Captured frame is None.")

            # Sleep to match the desired frame rate
            time.sleep(1 / fps)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        logger.error(f"Exception in main loop: {e}")
        traceback.print_exc()
    finally:
        # Clean up PiCamera2
        picam2.stop()
        logger.info("PiCamera2 stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()

