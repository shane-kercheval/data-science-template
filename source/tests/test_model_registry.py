import pytest
import os
import mlflow
import mlflow.exceptions
import pandas as pd

from helpsk.validation import dataframes_match
from source.service.model_registry import MLStage, Tracker, ModelRegistry, Experiment, Run


@pytest.mark.usefixtures('start_ml_server')
def test_services(tracking_uri, data_split):
    experiment_name = 'test_experiment'
    metric = 'roc_auc'
    x_train, x_test, y_train, y_test = data_split

    # experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment(experiment_name=experiment_name)
    assert exp is None

    versions = registry.get_model_latest_verisons(model_name='credit_model')
    assert len(versions) == 0
    version = registry.get_production_model(model_name='credit_model')
    assert version is None

    def read_text(file_path):
        with open(file_path, 'r') as handle:
            return handle.read()

    # tracker = Tracker(
    #     tracking_uri=tracking_uri,
    #     experiment_name=experiment_name,
    #     tags=dict(type='BayesSearchCV')
    # )
    tracker = registry.track_experiment(
        experiment_name=experiment_name,
        tags=dict(type='BayesSearchCV')
    )

    def run_experiment(fake_metric: str, fake_params: dict, fake_value: str, fake_name: str):
        with tracker:
            tracker.log_text(fake_value, fake_name)
            assert not os.path.isdir(Tracker.TEMP_DIR)
            tracker.log_model(model='mock')
            tracker.log_metric(metric=metric, metric_value=fake_metric)
            tracker.log_params(params=fake_params)
            tracker.log_pickle(obj=x_train, file_name='x_train.pkl')
            tracker.log_pickle(obj=x_test, file_name='x_test.pkl')
            tracker.log_pickle(obj=y_train, file_name='y_train.pkl')
            tracker.log_pickle(obj=y_test, file_name='y_test.pkl')

    ####
    # First Experiment
    ####
    run_experiment(
        fake_metric=0.9,
        fake_params={'param1': 'value', 'param2': '2'},
        fake_value='run 1',
        fake_name='run 1 text'
    )

    assert tracker.elapsed_seconds > 0
    assert tracker.last_run_name is not None
    exp = registry.get_experiment(experiment_name=experiment_name)
    assert exp is None  # value is cached
    exp = Experiment.load(experiment_name=experiment_name, registry=registry)
    assert exp is None

    registry.clear_cache()

    exp = registry.get_experiment(experiment_name=experiment_name)
    assert exp is not None
    assert isinstance(exp, mlflow.entities.experiment.Experiment)
    assert exp.experiment_id == '1'
    assert exp.name == experiment_name

    exp = Experiment.load(experiment_name=experiment_name, registry=registry)
    assert exp is not None
    assert exp.name == experiment_name
    assert exp.id == '1'
    assert exp.last_run.name == tracker.last_run_name
    assert exp.last_run.experiment_name == experiment_name
    assert exp.last_run.experiment_id == '1'
    assert exp.last_run.start_time is not None
    assert exp.last_run.end_time is not None
    assert exp.last_run.metrics == {'roc_auc': 0.9}
    assert exp.last_run.params == {'param1': 'value', 'param2': '2'}
    assert exp.last_run.tags['type'] == 'BayesSearchCV'
    logged_value = exp.last_run.download_artifact(artifact_name='run 1 text', read_from=read_text)
    assert logged_value == 'run 1'
    downloaded_x_train = exp.last_run.download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    runs = exp.runs
    assert len(runs) == 1
    assert runs[0].name == tracker.last_run_name
    assert runs[0].experiment_name == experiment_name
    assert runs[0].experiment_id == '1'
    assert runs[0].start_time is not None
    assert runs[0].end_time is not None
    assert runs[0].metrics == {'roc_auc': 0.9}
    assert runs[0].params == {'param1': 'value', 'param2': '2'}
    assert runs[0].tags['type'] == 'BayesSearchCV'
    logged_value = runs[0].download_artifact(artifact_name='run 1 text', read_from=read_text)
    assert logged_value == 'run 1'
    downloaded_x_train = runs[0].download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    run_from_name = Run.load(
        experiment_name=experiment_name,
        run_name=tracker.last_run_name,
        registry=registry
    )
    assert run_from_name.name == exp.last_run.name
    assert run_from_name.experiment_name == exp.last_run.experiment_name
    assert run_from_name.experiment_id == exp.last_run.experiment_id
    assert run_from_name.start_time == exp.last_run.start_time
    assert run_from_name.end_time == exp.last_run.end_time

    run_from_id = Run.load_from_id(
        experiment_name=experiment_name,
        run_id=exp.last_run.id,
        registry=registry
    )
    assert run_from_id.name == exp.last_run.name
    assert run_from_id.experiment_name == exp.last_run.experiment_name
    assert run_from_id.experiment_id == exp.last_run.experiment_id
    assert run_from_id.start_time == exp.last_run.start_time
    assert run_from_id.end_time == exp.last_run.end_time

    ####
    # test model registry and transition
    ####
    # at this point there are no models yet
    versions = registry.get_model_latest_verisons(model_name='credit_model')
    assert len(versions) == 0
    version = registry.get_production_model(model_name='credit_model')
    assert version is None
    # register model
    model_version = exp.last_run.register_model(model_name='credit_model')
    assert model_version.name == 'credit_model'
    assert model_version.version == '1'
    assert model_version.current_stage == 'None'
    # check that we can get model from registry
    versions = registry.get_model_latest_verisons(model_name='credit_model')
    assert len(versions) == 1
    assert versions[0].name == model_version.name
    assert versions[0].version == model_version.version
    assert versions[0].current_stage == model_version.current_stage
    # there shouldn't be any production models
    version = registry.get_production_model(model_name='credit_model')
    assert version is None
    # transition newly registered model into production
    new_version = registry.transition_model_to_stage(
        model_name='credit_model',
        model_version=model_version.version,
        to_stage=MLStage.PRODUCTION
    )
    assert new_version.name == model_version.name
    assert new_version.version == model_version.version
    assert new_version.current_stage == MLStage.PRODUCTION.value

    version = registry.get_production_model(model_name='credit_model')
    assert version.name == model_version.name
    assert version.version == model_version.version
    assert version.current_stage == MLStage.PRODUCTION.value

    ####
    # Second Experiment
    ####
    first_exp_seconds = tracker.elapsed_seconds
    first_exp_run_name = tracker.last_run_name
    first_exp_start_time = exp.last_run.start_time
    first_exp_end_time = exp.last_run.end_time

    run_experiment(
        fake_metric=0.8,
        fake_params={'param1': 'value2', 'param2': '3'},
        fake_value='run 2',
        fake_name='run 2 text'
    )

    assert tracker.elapsed_seconds > 0
    assert tracker.elapsed_seconds != first_exp_seconds
    assert tracker.last_run_name is not None
    assert tracker.last_run_name != first_exp_run_name

    # same experiment
    exp = Experiment.load(experiment_name=experiment_name, registry=registry)
    assert exp is not None
    assert exp.name == experiment_name
    assert exp.id == '1'

    # cache isn't cleared
    assert exp.last_run.name == first_exp_run_name
    assert len(exp.runs) == 1

    registry.clear_cache()
    assert exp.last_run.name == tracker.last_run_name
    assert len(exp.runs) == 2
    assert exp.last_run.experiment_name == experiment_name
    assert exp.last_run.experiment_id == '1'
    assert exp.last_run.start_time is not None
    assert exp.last_run.start_time > first_exp_start_time
    assert exp.last_run.start_time == max(x.start_time for x in exp.runs)
    assert exp.last_run.end_time is not None
    assert exp.last_run.end_time > first_exp_end_time
    assert exp.last_run.metrics == {'roc_auc': 0.8}
    assert exp.last_run.params == {'param1': 'value2', 'param2': '3'}
    assert exp.last_run.tags['type'] == 'BayesSearchCV'
    logged_value = exp.last_run.download_artifact(artifact_name='run 2 text', read_from=read_text)
    assert logged_value == 'run 2'
    downloaded_x_train = exp.last_run.download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    runs = exp.runs
    assert len(runs) == 2
    assert runs[0].name == tracker.last_run_name
    assert runs[0].experiment_name == experiment_name
    assert runs[0].experiment_id == '1'
    assert runs[0].start_time is not None
    assert runs[0].end_time is not None
    assert runs[0].metrics == {'roc_auc': 0.8}
    assert runs[0].params == {'param1': 'value2', 'param2': '3'}
    assert runs[0].tags['type'] == 'BayesSearchCV'
    logged_value = runs[0].download_artifact(artifact_name='run 2 text', read_from=read_text)
    assert logged_value == 'run 2'
    downloaded_x_train = runs[0].download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    ####
    # test model registry and transition
    ####
    # there should be one model version from the last run
    versions = registry.get_model_latest_verisons(model_name='credit_model')
    assert len(versions) == 1
    assert versions[0].version == '1'

    # register model, check that the version is 2
    model_version = exp.last_run.register_model(model_name='credit_model')
    assert model_version.name == 'credit_model'
    assert model_version.version == '2'
    assert model_version.current_stage == 'None'

    # check that we can get model from registry
    versions = registry.get_model_latest_verisons(model_name='credit_model')
    assert len(versions) == 2
    assert versions[1].name == model_version.name
    assert versions[1].version == model_version.version
    assert versions[1].current_stage == model_version.current_stage

    # get the production model, check that the version is still 1
    prod_version = registry.get_production_model(model_name='credit_model')
    assert prod_version.name == model_version.name
    assert prod_version.version == '1'
    assert prod_version.current_stage == MLStage.PRODUCTION.value

    # transition production model to archived
    new_prod_version = registry.transition_model_to_stage(
        model_name=prod_version.name,
        model_version=prod_version.version,
        to_stage=MLStage.ARCHIVED
    )
    assert prod_version.version == '1'
    assert new_prod_version.name == prod_version.name
    assert new_prod_version.version == prod_version.version
    assert new_prod_version.current_stage == 'Archived'

    # transition newly registered model into production
    assert model_version.version == '2'
    new_version = registry.transition_model_to_stage(
        model_name=model_version.name,
        model_version=model_version.version,
        to_stage=MLStage.PRODUCTION
    )
    assert new_version.name == model_version.name
    assert new_version.version == model_version.version
    assert new_version.current_stage == MLStage.PRODUCTION.value

    version = registry.get_production_model(model_name='credit_model')
    assert model_version.version == '2'
    assert version.name == model_version.name
    assert version.version == model_version.version
    assert version.current_stage == MLStage.PRODUCTION.value
