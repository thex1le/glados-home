# built ins
from json import dumps
from threading import Thread
from time import sleep

# 3rd party
import zmq
from glados_modules import GlogConfig


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = GlogConfig.setup_logger(name=self.__name__)
        self.configfile = configfile["DEFAULT"]
        ip = self.configfile["ZMQSenderAddress"]
        port = self.configfile["ZMQSenderPort"]
        self.context = zmq.Context()
        self.client_address = f"tcp://{ip}:{port}"
        self.logger.debug(f"Data Sender listening on {self.client_address}")
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(self.client_address)
        self.stop = False
        self.data = list()

    def stop_thread(self):
        self.logger.debug("ZMQ Sending Thread Stop Called")
        self.stop = True

    def send_data(self, data, json_send=True):
        if json_send is True:
            self.logger.debug("Sending JSON data")
            self.data.append(dumps(data))
        if json_send is False:
            self.logger.debug("Sending PICKLE Data")
            self.data.append(dumps(data))

    def run(self):
        msg = "Data Sending Loop Started"
        print(msg)
        self.logger.debug(msg)
        while self.stop is False:
            try:
                data = self.data.pop(0)
                if type(data) is str:
                    self.socket.send_string(data)
                else:
                    self.socket.send(data)
            except IndexError:
                pass
            sleep(.1)
        self.socket.close()
        self.context.term()


class DataRecv(Thread):
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = GlogConfig.setup_logger(self.__name__)
        self.config = configfile["DEFAULT"]
        ip = self.config["ZMQListenAddress"]
        port = self.config["ZMQSenderPort"]
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # Create a PULL socket
        self.client_address = f"tcp://{ip}:{port}"
        self.socket.bind(self.client_address)  # Bind to the TCP port 5555
        self.logger.debug(f"Data Sender listening on {self.client_address}")
        self.data = None
        self.stop = False

    def get_data(self, blocking=False):
        if blocking is True:
            while self.data is None:
                sleep(.1)
        data = self.data
        self.data = None
        self.logger.debug(f"Data from zmq returned f{data}")
        return data

    def stop_thread(self):
        self.logger.debug("ZMQ Receiving Thread Stop Called")
        self.stop = True

    def run(self):
        while self.stop is False:
            # TODO check if there is a timeout here or it will never close on exit?
            self.data = self.socket.recv()  # Receive the data as a JSON string
            msg = "Got data from ZMQ Listener"
            print(msg)
            self.logger.debug(msg)
        self.socket.close()
        self.context.term()

