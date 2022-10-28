# -*- coding: utf-8 -*-
import logging
from functools import wraps

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def log_function_name(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f"Running fuction: {f.__name__}")
        return f(*args, **kwargs)

    return wrapper
