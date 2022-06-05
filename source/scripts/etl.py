"""
This file contains the logic for extracting and transforming the project data.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

sys.path.append(os.getcwd())
from source.library.utilities import log_info, log_func  # noqa


def extract(output_directory: str):
    """This function downloads the credit data from openml.org."""
    log_info("Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    log_info(f"Credit data downloaded with {credit_data.shape[0]} rows and {credit_data.shape[1]} columns.")

    output_file = os.path.join(output_directory, 'credit.pkl')
    log_info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


def transform(input_directory: str, output_directory: str):
    """
    This function transforms the credit data.

    Args:
        input_directory: the directory to find credit.pkl
        output_directory: the directory to save the modified credit.pkl to
    """
    log_func("transform", params=dict(
        input_directory=input_directory,
        output_directory=output_directory,
    ))

    credit_data = pd.read_pickle(os.path.join(input_directory, 'credit.pkl'))
    log_info("Transforming Data")

    # Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0

    log_info("Done processing credit data.")

    output_file = os.path.join(output_directory, 'credit.pkl')
    log_info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)
