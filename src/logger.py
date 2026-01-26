import logging
import sys
import os


def setup_logger(log_file_path):
    """
    sets up a logger that writes to both console and a file
    - args:
        log_file_path (str): the path where the log file will be created
    - returns:
        logger: a configured logger object
    """
    logger = logging.getLogger("summary_engine")
    # set to debug at highest level, since we need DEBUG for the files
    logger.setLevel(logging.DEBUG)
    # prevent duplicate logs if this logger is run multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    # create formatters
    # file detailed: [Time] [Level] Message
    file_formatter = logging.Formatter(
        # We add .{msecs:03.0f} right after {asctime}
        fmt="[{asctime}.{msecs:03.0f}] [{levelname}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{"
    )
    # console minimal
    console_formatter = logging.Formatter(
        fmt="{message}",
        style="{"
    )
    # create handlers
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # file gets everything
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # console only gets INFO and above (hides DEBUG noise)
    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
