import argparse
import os
from datetime import datetime
import sys
import logging


def parse_range(value: str) -> tuple[int, int]:
    """
    - serveral cli entry points receive scene range with start end values
    - parse 'start,end' string into tuple for argparse
    """
    try:
        parts = value.split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(f"must be start,end format (e.g. 0,10), got: {value}")


def init_logger(operation_type: str, debug_dir: str, book_name: str) -> logging.Logger:
    """
    - setup logging for modules with params from config.py -> handler for console and logfile
    - create dir for logfile if necessary; construct file path
    - book_name caller must provide the name/prefix for the log file - no path, no suffix!
    """
    os.makedirs(debug_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    logfile_path = os.path.join(debug_dir, f"{book_name}_{operation_type}_{ts}.log")
    # setup logger
    logger = logging.getLogger(operation_type)
    # set to debug at highest level
    logger.setLevel(logging.DEBUG)
    # guard to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    # create formatters: file detailed: [Time] [Level] Message; console minimal
    file_formatter = logging.Formatter(
        # We add .{msecs:03.0f} right after {asctime}
        fmt="[{asctime}.{msecs:03.0f}] [{levelname}] {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{"
    )
    console_formatter = logging.Formatter(
        fmt="{message}",
        style="{"
    )
    # file_handler
    file_handler = logging.FileHandler(logfile_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # file gets everything
    # console_handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # console only INFO and above (hide DEBUG noise)
    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
