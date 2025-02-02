import logging
from os import path, makedirs
from glados_modules.GLaDosEnums import LoggingEnums


def setup_logger(
        name: str = __name__,
        level: int = logging.DEBUG,
        log_dir: str = LoggingEnums.LOG_FOLDER_DEFAULT_NAME.value,
        file_logging: int = LoggingEnums.LOG_DEBUG.value,
        console_logging: int = LoggingEnums.LOG_DEBUG.value
) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.

    This function ensures that log messages are saved to a file while also being
    displayed in the console. The log file is stored in the specified log directory,
    and the log level can be adjusted for both file and console handlers.

    Args:
        name (str): The name of the logger. Defaults to `__name__`.
        level (int): The overall logging level for the logger. Defaults to `logging.DEBUG`.
        log_dir (str): The directory where log files are stored. Defaults to `LoggingEnums.LOG_FOLDER_DEFAULT_NAME.value`.
        file_logging (int): The logging level for file output. Defaults to `LoggingEnums.LOG_DEBUG.value`.
        console_logging (int): The logging level for console output. Defaults to `LoggingEnums.LOG_DEBUG.value`.

    Returns:
        logging.Logger: A configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers from being added
    if not logger.hasHandlers():
        # Ensure log directory exists
        if not path.exists(log_dir):
            makedirs(log_dir)

        # Define log file path with the correct extension
        log_file = path.join(log_dir, f"{name}{LoggingEnums.LOG_FILE_TYPE.value}")

        # File handler setup
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_logging)

        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_logging)

        # Define log format
        formatter = logging.Formatter(LoggingEnums.LOG_FORMAT.value)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
