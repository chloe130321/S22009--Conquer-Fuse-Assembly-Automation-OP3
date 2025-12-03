import logging
import os
import time

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# Dictionary to store loggers so they don't get recreated
loggers = {}

def setup_logger(name, level=logging.INFO):
    global loggers

    if name in loggers:
        return loggers[name]

    log_path = os.path.join(r"C:\2-3_2-6", "log")
    os.makedirs(log_path, exist_ok=True)

    log_time = time.strftime("%Y-%m-%d")
    filename = os.path.join(log_path, f"{log_time}_{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate file handlers
    already_exists = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(filename)
        for h in logger.handlers
    )

    if not already_exists:
        file_handler = logging.FileHandler(filename, encoding="utf-8", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 🔹 Store file handler so we can close it later (optional, if needed)
        logger._file_handler = file_handler

    # Also show logs in console
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    loggers[name] = logger
    return logger


def loginfo(name, message):
    """Write log message while ensuring proper logger reuse."""
    logger = setup_logger(name)
    logger.info(message)
