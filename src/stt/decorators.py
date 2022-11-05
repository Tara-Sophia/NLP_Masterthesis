# -*- coding: utf-8 -*-
import logging
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

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


def log_function_name(f: Callable[P, Callable]) -> Callable[P, R]:
    """
    Decorator to log the name of the function

    Parameters
    ----------
    f : Callable[P, Callable]
        Function to decorate

    Returns
    -------
    Callable[P, R]
        Decorated function
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """
        Logging the function name

        Returns
        -------
        R
            Original function return value
        """
        logging.info(f"Running STT fuction: {f.__name__}")
        return f(*args, **kwargs)

    return wrapper
