# -*- coding: utf-8 -*-
import logging
from functools import wraps

GREEN_LOGGER = "\x1b[32m"
RESET_LOGGER = "\x1b[0m"
FORMAT = "%(asctime)s.%(msecs)03d  - %(message)s"

# create logger with 'spam_application'
logger = logging.getLogger("STT")

logging.basicConfig(
    format=GREEN_LOGGER + FORMAT + RESET_LOGGER,
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def log_function_name(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f"Running STT fuction: {f.__name__}")
        return f(*args, **kwargs)

    return wrapper
