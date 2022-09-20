"""
This file contains the logic for extracting and transforming the project data.
"""
import os
import warnings
import logging

import numpy as np
import pandas as pd

from helpsk.logging import log_function_call, log_timer
from sklearn.datasets import fetch_openml


@log_function_call
@log_timer
def extract(output_directory: str):
    """This function downloads the credit data from openml.org."""
    logging.info("Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    logging.info(
        f"Credit data downloaded with {credit_data.shape[0]} "
        f"rows and {credit_data.shape[1]} columns."
    )
    output_file = os.path.join(output_directory, 'credit.pkl')
    logging.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


@log_function_call
@log_timer
def transform(input_directory: str, output_directory: str):
    """
    This function transforms the credit data.

    Args:
        input_directory: the directory to find credit.pkl
        output_directory: the directory to save the modified credit.pkl to
    """
    credit_data = pd.read_pickle(os.path.join(input_directory, 'credit.pkl'))
    logging.info("Transforming Data")

    # Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0

    logging.info("Done processing credit data.")

    output_file = os.path.join(output_directory, 'credit.pkl')
    logging.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)
