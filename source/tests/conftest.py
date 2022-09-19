"""This file defines test fixtures for pytest unit-tests."""
import pytest
import os
import shutil
import subprocess
import requests
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import logging.config

logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'source/tests/test_files/log.log'},
    disable_existing_loggers=False
)


@pytest.fixture(scope='session')
def credit_data() -> pd.DataFrame:
    return pd.read_pickle('data/processed/credit.pkl')


@pytest.fixture(scope='session')
def data_split(credit_data) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')
    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    # keep random_state the same across experiments to compare apples to apples
    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full,
        test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


@pytest.fixture(scope='session')
def tracking_uri() -> str:
    return "http://0.0.0.0:1235"


@pytest.fixture(scope='module')
def ml_server_directory(tmpdir_factory):
    test_dir = str(tmpdir_factory.mktemp('ml_server'))
    logging.info(f'ml_server_directory: creating {test_dir}')
    # return test_dir
    try:
        yield test_dir
    finally:
        logging.info(f'ml_server_directory: removing {test_dir}')
        shutil.rmtree(test_dir)


@pytest.fixture(scope='module')
def start_ml_server(ml_server_directory, tracking_uri):
    os.system("lsof -t -i:1235 | xargs -r kill")
    command = f"""
        mlflow server
            --backend-store-uri sqlite:///{os.path.join(ml_server_directory, 'mlflow.db')}
            --default-artifact-root {os.path.join(ml_server_directory, 'mlflow-artifact-root')}
            --host 0.0.0.0
            --port 1235
    """
    daemon = subprocess.Popen(command.split())
    logging.info(f'start_ml_server: starting server; process {daemon.pid}')
    wait_for_service_to_start(tracking_uri)
    yield
    logging.info(f'start_ml_server: stopping server; process {daemon.pid}')
    daemon.kill()
    os.system("lsof -t -i:1235 | xargs -r kill")


def wait_for_service_to_start(url: str):
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            return requests.get(url)
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    pytest.fail(f"'{url}' never started")
