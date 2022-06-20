"""
This file provides general helper functions such as logging and saving/pickling objects.
"""
import datetime
from functools import wraps
import logging
import logging.config
import os
from pathlib import Path
from typing import Callable, Union
import yaml

import pandas as pd
from helpsk.utility import to_pickle


def get_config(file_path: str = "source/config/config.yaml"):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def log_info(message: str):
    """
    Calls logging.info. Use this function rather than logging.info directly in case a production
    environment requires a different library/setup.

    Args:
        message: the message to log
    """
    logging.info(message)


def log_function_call(function: Callable) -> Callable:
    """
    This function should be used as a decorator to log the function name and paramters of the function when
    called.

    Args: function that is decorated
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        function.__name__
        if len(args) == 0 and len(kwargs) == 0:
            _log_function(function_name=function.__name__, params=None)
        else:
            parameters = dict()
            if len(args) > 0:
                parameters['args'] = args
            if len(kwargs) > 0:
                parameters.update(kwargs)
            _log_function(function_name=function.__name__, params=parameters)

        return function(*args, **kwargs)
    return wrapper


def _log_function(function_name: str, params: Union[dict, None] = None):
    """
    This function is meant to be used at the start of the calling function; calls log_info and passes the
    name of the function and optional parameter names/values.

    Args:
        func_name:
            the name of the function
        params:
            a dictionary containing the names of the function parameters (as dictionary keys) and the
            parameter values (as dictionary values).
    """
    log_info(f"FUNCTION: {function_name.upper()}")
    if params is not None:
        log_info("PARAMS:")
        for key, value in params.items():
            log_info(f"    {key}: {value}")


def dataframe_to_pickle(df: pd.DataFrame, output_directory: str, file_name: str) -> str:
    """
    This function takes a Pandas DataFrame and saves it as a pickled object to the directory with the file
    name specified. The output directory is created if it does not yet exist.

    Args:
        df: the Pandas DataFrame to pickle
        output_directory: the directory to save the pickled object
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    df.to_pickle(file_path)
    return file_path


def dataframe_to_csv(df: pd.DataFrame, output_directory: str, file_name: str) -> str:
    """
    This function takes a Pandas DataFrame and saves it as a csv file to the directory with the file
    name specified. The output directory is created if it does not yet exist.

    Args:
        df: the Pandas DataFrame to pickle
        output_directory: the directory to save the csv file
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    df.to_csv(file_path, index=False)
    return file_path


def object_to_pickle(obj: object, output_directory: str, file_name: str) -> str:
    """
    This function takes a generic object and saves it as a pickled object to the directory with the file
    name specified. The output directory is created if it does not yet exist.

    Args:
        obj: the object to pickle
        output_directory: the directory to save the pickled object
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    to_pickle(obj=obj, path=file_path)
    return file_path


class Timer:
    """
    This class provides way to time the duration of code within the context manager.
    """
    def __init__(self, message, include_message_at_finish=False):
        self._message = message
        self._include_message_at_finish = include_message_at_finish

    def __enter__(self):
        # log_info()(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Starting Timer: {self._message}\n"
        #              f"                          ======")
        # "2022-05-26-11:19:46 - INFO     | Finished (10.82 seconds)"
        logging.basicConfig()
        log_info(f'Timer Started: {self._message}')
        self._start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = ''
        if self._include_message_at_finish:
            message = self._message + " "

        log_info(f'Timer Finished: {message}({self._interval.total_seconds():.2f} seconds)')


def log_timer(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs):
        with Timer(f"FUNCTION={function.__module__}:{function.__name__}", include_message_at_finish=True):
            results = function(*args, **kwargs)
        return results

    return wrapper
