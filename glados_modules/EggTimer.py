from threading import Thread, Lock
import time

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class EggTimer(Thread):
    def __init__(self, duration_in_seconds, speak):
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__class__.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.duration = duration_in_seconds
        self.start_time = None
        self.is_running = False
        self.speak = speak
        self.lock = Lock()

    def timer_start(self):
        with self.lock:
            if not self.is_running:
                self.start_time = time.time()
                self.is_running = True
                msg = f"Egg timer started for {self.duration} seconds."
                self.logger.debug(msg)

    def stop(self):
        with self.lock:
            if self.is_running:
                elapsed_time = time.time() - self.start_time
                remaining_time = max(0, self.duration - elapsed_time)
                self.is_running = False
                msg = "Timer stopped. Remaining time: {:.2f} seconds.".format(remaining_time)
                self.speak(msg)

    def check_remaining_time(self):
        rtn = {"remain": 0, "complete": False}
        with self.lock:
            if self.is_running:
                elapsed_time = time.time() - self.start_time
                remaining_time = max(0, self.duration - elapsed_time)
                rtn["remain"] = remaining_time
                if remaining_time == 0:
                    rtn["remain"] = 0
                    rtn["complete"] = True
                    msg = "Your Timer is complete"
                    self.logger.debug(msg)
                    self.speak(msg)
            else:
                rtn["remain"] = 0
                rtn["complete"] = True
        return rtn

    def run(self):
        self.timer_start()
        while True:
            r = self.check_remaining_time()
            self.logger.debug(r)
            if r["complete"] is True:
                break
            time.sleep(0.2)
