from multiprocessing.sharedctypes import Value
import pytest
import subprocess
import requests
import time

import pandas as pd


@pytest.fixture
def credit_data():
    return pd.read_pickle('data/processed/credit.pkl')


@pytest.fixture(scope='session')
def start_ml_server():
    command = """
        mlflow server
            --backend-store-uri sqlite:///mlflow.db
            --default-artifact-root ./mlflow-artifact-root
            --host 0.0.0.0
            --port 1235
    """
    daemon = subprocess.Popen(command.split())
    wait_for_service_to_start("http://0.0.0.0:1235")
    yield
    daemon.kill()


def wait_for_service_to_start(url: str):
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            return requests.get(url)
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    pytest.fail(f"'{url}' never started")
