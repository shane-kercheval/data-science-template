import os
import sys
import warnings

import numpy as np
from helpsk.utility import read_pickle, to_pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

sys.path.append(os.getcwd())
from source.library.utilities import log_info  # noqa


def extract():
    log_info(f"Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    log_info(f"Credit data downloaded with {credit_data.shape[0]} rows and {credit_data.shape[1]} columns.")
    
    # Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0
    
    log_info(f"Done processing credit data.")

    output_file = 'artifacts/data/raw/credit.pkl'
    log_info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


def transform():
    log_info(f"Transforming Data")


def create_training_test(input_directory, output_directory):
    log_info(f"Splitting training & test datasets")

    credit_data = read_pickle(os.path.join(input_directory, 'credit.pkl'))

    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')

    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    file_name = os.path.join(output_directory, 'x_train.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(x_train, file_name)

    file_name = os.path.join(output_directory, 'x_test.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(x_test, file_name)

    file_name = os.path.join(output_directory, 'y_train.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(y_train, file_name)

    file_name = os.path.join(output_directory, 'y_test.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(y_test, file_name)
