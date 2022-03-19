import os
import sys
import warnings

import click
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from helpsk.utility import read_pickle, to_pickle

sys.path.append(os.getcwd())
from source.library.utilities import get_logger  # noqa


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    logger = get_logger()
    logger.info(f"Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    logger.info(f"Credit data downloaded with {credit_data.shape[0]} rows and {credit_data.shape[1]} columns.")
    
    # Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0
    
    logger.info(f"Done processing credit data.")

    output_file = 'artifacts/data/raw/credit.pkl'
    logger.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


@main.command()
def transform():
    logger = get_logger()
    logger.info(f"Transforming Data")


@main.command()
def create_training_test():
    logger = get_logger()
    logger.info(f"Splitting training & test datasets")

    credit_data = read_pickle('artifacts/data/raw/credit.pkl')

    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')

    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    file_name = 'artifacts/data/processed/x_train.pkl'
    logger.info(f"Creating `{file_name}`")
    to_pickle(x_train, file_name)

    file_name = 'artifacts/data/processed/x_test.pkl'
    logger.info(f"Creating `{file_name}`")
    to_pickle(x_test, file_name)

    file_name = 'artifacts/data/processed/y_train.pkl'
    logger.info(f"Creating `{file_name}`")
    to_pickle(y_train, file_name)

    file_name = 'artifacts/data/processed/y_test.pkl'
    logger.info(f"Creating `{file_name}`")
    to_pickle(y_test, file_name)


if __name__ == '__main__':
    main()
