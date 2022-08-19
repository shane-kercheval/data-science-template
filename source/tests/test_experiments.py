import pytest
import requests
from mlflow.tracking import MlflowClient


@pytest.mark.usefixtures('start_ml_server')
def test_experiment(credit_data):
    assert len(credit_data) > 0
    assert len(credit_data) > 0
    requests.get("http://0.0.0.0:1235")
    client = MlflowClient(tracking_uri="http://0.0.0.0:1235")
    # experiment = client.get_experiment_by_name('credit')
    # value = 10


def test_2():
    assert True
