import logging
from os import path, makedirs


# TODO make this take levels from config file
def setup_logger(name=__name__, level=logging.DEBUG, log_dir="logs",
                 file_logging=logging.DEBUG, console_logging=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not path.exists(log_dir):
        makedirs(log_dir)
    fh = logging.FileHandler(path.join(log_dir, name))
    fh.setLevel(file_logging)
    ch = logging.StreamHandler()
    ch.setLevel(console_logging)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.getEffectiveLevel()
    return logger
