# built ins
from pickle import dumps, loads
from threading import Thread
from time import sleep
from queue import Queue, Empty

# 3rd party
import zmq
from glados_modules import GlogConfig


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, configfile, location):
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ = location
        self.logger = GlogConfig.setup_logger(name=self.__name__)
        self.configfile = configfile["DEFAULT"]
        ip = self.configfile["ZMQSenderAddress"]
        port = self.configfile["ZMQSenderPort"]
        self.context = zmq.Context()
        self.client_address = f"tcp://{ip}:{port}"
        self.logger.debug(f"Data Sender connecting too {self.client_address}")
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(self.client_address)
        self.stop = False
        self.queue = Queue()

    def stop_thread(self):
        self.logger.debug("ZMQ Sending Thread Stop Called")
        self.stop = True

    def send_data(self, data):
        self.logger.debug("Sending PICKLE Data")
        self.queue.put(dumps(data))

    def run(self):
        msg = "Data Sending Loop Started"
        self.logger.debug(msg)
        while self.stop is False:
            try:
                data = self.queue.get()
                print(f"Sending Queue backlog is {self.queue.qsize()}")
                self.socket.send(data)
            except zmq.error.ZMQError as e:
                self.logger.error(f"ZMQ Error: {e}")
                break  # Break out of the loop if there is a ZMQ error
            except Empty:
                self.logger.debug("Empty Queue")
                sleep(.1)
                pass
        self.socket.close()
        self.context.term()


class DataRecv(Thread):
    def __init__(self, configfile, location):
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = location
        self.logger = GlogConfig.setup_logger(self.__name__)
        self.config = configfile["DEFAULT"]
        ip = self.config["ZMQListenAddress"]
        port = self.config["ZMQSenderPort"]
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.client_address = f"tcp://{ip}:{port}"
        self.socket.bind(self.client_address)
        self.logger.debug(f"Data Receiver listening on {self.client_address}")
        self.queue = Queue()  # Thread-safe queue for storing data
        self.stop = False

    def get_data_from_queue(self, blocking: bool = True, timeout: bool = None):
        """
        Pulls data from the queue. This method can be accessed by another thread.
        :param blocking: If True, block until an item is available. Otherwise, return immediately.
        :param timeout: How long to block before raising an Empty exception if no item is available.
        :return: The data item, or None if not blocking and the queue is empty.
        """
        try:
            data = self.queue.get(block=blocking, timeout=timeout)
            return loads(data)  # Unpickle/deserialize the data
        except Empty:
            self.logger.debug("Queue is empty, no data to return.")
            return None

    def stop_thread(self) -> None:
        """
        Nicely Stop the thread
        """
        self.logger.debug("ZMQ Receiving Thread Stop Called")
        self.stop = True

    def run(self):
        while not self.stop:
            try:
                data = self.socket.recv()  # Receive data from the socket
                self.queue.put(data)  # Put the data into the queue
                print(f"Receive Queue Backlog is {self.queue.qsize()}")
                msg = "Got data from ZMQ Listener and pushed to queue"
                self.logger.debug(msg)
            except zmq.error.ZMQError as e:
                self.logger.error(f"ZMQ Error: {e}")
                break  # Break out of the loop if there is a ZMQ error
        self.socket.close()
        self.context.term()
