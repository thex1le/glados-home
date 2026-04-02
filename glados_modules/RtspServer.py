#!/usr/bin/env python3
from threading import Thread, Lock
from time import time
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib
import cv2

# glados modules
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import CameraEnum, LoggingEnums


class RtspSystem(GstRtspServer.RTSPMediaFactory):
    def __init__(self, cam_x, cam_y, fps=30, **properties):
        super(RtspSystem, self).__init__(**properties)
        self.cam_x = int(cam_x)
        self.cam_y = int(cam_y)
        self.data = None
        self.data_lock = Lock()
        self.number_frames = 0
        self._last_push_time = 0
        self._push_count = 0
        # TODO figure out how to do hardware encoding
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=0 name=pay0 pt=96'.format(self.cam_x, self.cam_y, fps)

    def send_data(self, data):
        with self.data_lock:
            self.data = data

    def start(self):
        t = Thread(target=self._thread_rtsp)
        t.start()

    def _thread_rtsp(self):
        loop = GLib.MainLoop()
        loop.run()

    def on_need_data(self, src, length):
        # Copy frame reference under lock, then push outside the lock.
        # push-buffer can block if the encoder queue is full — holding
        # data_lock during that would deadlock the camera capture loop.
        with self.data_lock:
            frame = self.data
        if frame is not None:
            t0 = time()
            retval = src.emit('push-buffer', Gst.Buffer.new_wrapped(frame.tobytes()))
            push_time = time() - t0
            if push_time > 1.0:
                print(f"WARNING: push-buffer took {push_time:.2f}s")
            if retval != Gst.FlowReturn.OK:
                print(f"push-buffer returned {retval}")
            self._push_count += 1
            now = time()
            if now - self._last_push_time >= 10.0:
                elapsed = now - self._last_push_time if self._last_push_time > 0 else 1
                print(f"RTSP push: {self._push_count / elapsed:.1f} FPS")
                self._push_count = 0
                self._last_push_time = now

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.set_property("emit-signals", True)
        appsrc.connect('need-data', self.on_need_data)


class RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, cam_configs, port=8554, **properties):
        super(RTSPServer, self).__init__(**properties)
        self.logger = setup_logger(name=self.__class__.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.set_service(str(port))
        Gst.init(None)
        self.factories = {}
        self._initialize_factories(cam_configs)
        self.attach(None)
        self.logger.debug(f"RTSP Server Started on port {port}")

    def path_fixup(self, fpath: str) -> str:
        rtr = fpath
        if fpath[0] != '/':
            # append a / if needed
            rtr = f"/{fpath}"
        return rtr

    def _initialize_factories(self, cam_configs):
        mount_points = self.get_mount_points()
        for factory_path in cam_configs.keys():
            (cam_x, cam_y) = cam_configs[factory_path][CameraEnum.MSG_RESOLUTION.value]
            rtsp_system = RtspSystem(cam_x, cam_y, int(cam_configs[factory_path][CameraEnum.MSG_FPS.value]))
            rtsp_system.set_shared(True)
            factory_path = self.path_fixup(factory_path)
            mount_points.add_factory(factory_path, rtsp_system)
            rtsp_system.start()
            self.factories[factory_path] = rtsp_system
            self.logger.debug(f"Factory {factory_path} added with resolution {cam_x}x{cam_y}")

    def send_data(self, factory_path, data):
        factory_path = self.path_fixup(factory_path)
        self.logger.debug(f"Send Data called for factory {factory_path}")
        if factory_path in self.factories:
            self.logger.debug("Correct factory found")
            rtsp_system = self.factories[factory_path]
            data = cv2.resize(data, (rtsp_system.cam_x, rtsp_system.cam_y))
            self.logger.debug(f"Sending data on {factory_path}")
            rtsp_system.send_data(data)
        else:
            self.logger.error(f"No factory found for path: {factory_path}")


# Example usage:
if __name__ == "__main__":
    test_cam_configs = {
        "/camera1": {"resolution": (640, 480), "fps": 30},
        "/camera2": {"resolution": (1280, 720), "fps": 30},
        "/camera3": {"resolution": (1920, 1080), "fps": 30}
    }
    server = RTSPServer(test_cam_configs)
    # Simulate sending data to cameras
    # server.send_data("/camera1", frame1)
    # server.send_data("/camera2", frame2)
    # server.send_data("/camera3", frame3)
