import pytest

from source.domain.experiment import run_bayesian_search
from source.service.model_registry import ModelRegistry

@pytest.mark.usefixtures('start_ml_server')
def test_experiment(data_split, tracking_uri):
    experiment_name = 'test_experiment'
    model_name = 'credit_model'
    score = 'roc_auc'
    x_train, x_test, y_train, y_test = data_split

    # check that experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment_by_name(exp_name=experiment_name)
    assert exp is None
    exp = registry.get_experiment_by_id(exp_id='1')
    assert exp is None

    # run_bayesian_search(
    #     x_train=x_train,
    #     x_test=x_test,
    #     y_train=y_train,
    #     y_test=y_test,
    #     tracking_uri=tracking_uri,
    #     experiment_name=experiment_name,
    #     model_name=model_name,
    #     score=score,
    #     n_iterations=n_iterations,
    #     n_splits=n_splits,
    #     n_repeats=n_repeats,
    #     random_state=random_state,
    #     tags=tags,
    # )
