import pytest
from helpsk.sklearn_eval import MLExperimentResults

from source.domain.experiment import run_bayesian_search
from source.service.model_registry import ModelRegistry, MLStage


@pytest.mark.usefixtures('start_ml_server')
def test_experiment(data_split, tracking_uri):
    x_train, x_test, y_train, y_test = data_split
    experiment_name = 'test_experiment'
    model_name = 'credit_model'
    score = 'roc_auc'
    n_iterations = 1
    n_folds = 3
    n_repeats = 1
    random_state = 3
    tags = dict(type='BayesSearchCV')

    # check that experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment_by_name(exp_name=experiment_name)
    assert exp is None
    exp = registry.get_experiment_by_id(exp_id='1')
    assert exp is None

    put_in_prod, tracker = run_bayesian_search(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        model_name=model_name,
        score=score,
        n_iterations=n_iterations,
        n_folds=n_folds,
        n_repeats=n_repeats,
        random_state=random_state,
        tags=tags,
    )
    assert put_in_prod  # first model should be put into production
    assert tracker is not None
    # since the model was registered, the model_version should be set
    assert tracker.last_run.model_version is not None
    assert tracker.last_run.model_version.version == '1'
    assert tracker.last_run.run_id == tracker.last_run.model_version.run_id
    assert tracker.last_run.model_version.name == model_name
    assert tracker.last_run.model_version.current_stage == MLStage.PRODUCTION.value

    results = tracker.last_run.download_artifact(
        artifact_name='experiment.yaml',
        read_from=MLExperimentResults.from_yaml_file
    )
    assert tracker.last_run.metrics[results.score_names[0]] == results.best_score
