import os
import pickle
import logging
import logging.config
from pathlib import Path
import click
from sklearn.datasets import fetch_openml
import numpy as np

import warnings

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger(__name__)
project_directory = Path(__file__).resolve().parents[1]


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    logger.info(f"Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    logger.info(f"Credit data downloaded with {credit_data.shape[0]} rows and {credit_data.shape[1]} columns.")
    
    ## Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0
    
    logger.info(f"Done processing credit data.")

    output_file = os.path.join(project_directory, 'data/raw/credit_data.pkl')
    logger.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


@main.command()
def transform():
    logger.info(f"Transforming Data")



if __name__ == '__main__':
    main()
