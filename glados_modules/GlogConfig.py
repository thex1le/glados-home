import logging
from os import path, makedirs


# TODO make this take levels from config file
def setup_logger(name=__name__, level=logging.DEBUG, log_dir="logs",
                 file_logging=logging.DEBUG, console_logging=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Check if the logger already has handlers to avoid adding them multiple times
    if not logger.hasHandlers():
        if not path.exists(log_dir):
            makedirs(log_dir)
        # Ensure log file has a .log extension
        log_file = path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(file_logging)
        ch = logging.StreamHandler()
        ch.setLevel(console_logging)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
