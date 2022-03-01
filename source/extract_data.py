import os
import pickle
import logging
import logging.config
from pathlib import Path
from sklearn.datasets import fetch_openml
import numpy as np

def extract_data(project_directory, logger):
    logger.info(f"Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    print(credit_data.shape)
    logger.info(f"Credit data downloaded with {len(credit_data)} rows.")
    
    ## Create Missing Values
    credit_data['duration'].iloc[0:46] = np.nan
    credit_data['checking_status'].iloc[25:75] = np.nan
    credit_data['credit_amount'].iloc[10:54] = 0
    logger.info(f"Done processing credit data.")

    output_file = os.path.join(project_directory, 'data/raw/credit_data.pkl')
    logger.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)

if __name__ == '__main__':
    logging.config.fileConfig("config/logging/local.conf")
    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[1]
    extract_data(project_directory=project_dir, logger=logger)
