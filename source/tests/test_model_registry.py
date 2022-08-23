import pytest
import os
import pandas as pd

from helpsk.validation import dataframes_match
from source.service.model_registry import MLStage, Tracker, ModelRegistry


@pytest.mark.usefixtures('start_ml_server')
def test_registry(tracking_uri, data_split):
    experiment_name = 'test_experiment'
    model_name = 'credit_model'
    metric = 'roc_auc'
    x_train, x_test, y_train, y_test = data_split

    # experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment_by_name(exp_name=experiment_name)
    assert exp is None
    exp = registry.get_experiment_by_id(exp_id='1')
    assert exp is None

    run = registry.get_run_by_name(exp_name=experiment_name, run_name='not exists')
    assert run is None
    run = registry.get_run_by_id(run_id='not exists')
    assert run is None

    versions = registry.get_model_latest_verisons(model_name=model_name)
    assert len(versions) == 0

    version = registry.get_production_model(model_name=model_name)
    assert version is None

    run = registry.get_production_run(model_name=model_name)
    assert run is None

    artifact = registry.download_artifact(
        run_id='id',
        artifact_name='art',
        read_from=pd.read_pickle
    )
    assert artifact is None

    def read_text(file_path):
        with open(file_path, 'r') as handle:
            return handle.read()

    tracker = registry.track_experiment(
        exp_name=experiment_name,
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
    exp = registry.get_experiment_by_name(exp_name=experiment_name)
    assert exp is not None
    assert exp.name == experiment_name
    assert exp.exp_id == '1'
    assert exp.last_run.name == tracker.last_run_name
    assert exp.last_run.exp_name == experiment_name
    assert exp.last_run.exp_id == '1'
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
    assert runs[0].exp_name == experiment_name
    assert runs[0].exp_id == '1'
    assert runs[0].start_time is not None
    assert runs[0].end_time is not None
    assert runs[0].metrics == {'roc_auc': 0.9}
    assert runs[0].params == {'param1': 'value', 'param2': '2'}
    assert runs[0].tags['type'] == 'BayesSearchCV'
    logged_value = runs[0].download_artifact(artifact_name='run 1 text', read_from=read_text)
    assert logged_value == 'run 1'
    downloaded_x_train = runs[0].download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    run_by_name = registry.get_run_by_name(
        exp_name=experiment_name,
        run_name=tracker.last_run_name,
    )
    assert run_by_name.name == exp.last_run.name
    assert run_by_name.exp_name == exp.last_run.exp_name
    assert run_by_name.exp_id == exp.last_run.exp_id
    assert run_by_name.start_time == exp.last_run.start_time
    assert run_by_name.end_time == exp.last_run.end_time

    run_by_id = registry.get_run_by_id(run_id=exp.last_run.run_id)
    assert run_by_id.name == exp.last_run.name
    assert run_by_id.exp_name == exp.last_run.exp_name
    assert run_by_id.exp_id == exp.last_run.exp_id
    assert run_by_id.start_time == exp.last_run.start_time
    assert run_by_id.end_time == exp.last_run.end_time

    ####
    # test model registry and transition
    ####
    # at this point there are no models yet
    versions = registry.get_model_latest_verisons(model_name=model_name)
    assert len(versions) == 0
    version = registry.get_production_model(model_name=model_name)
    assert version is None
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run is None

    # register model
    assert exp.last_run.model_version is None
    assert exp.runs[0].model_version is None
    model_version = exp.last_run.register_model(model_name=model_name)
    assert exp.last_run.model_version.version == '1'
    assert exp.runs[0].model_version.version == '1'
    assert model_version.name == model_name
    assert model_version.version == '1'
    assert model_version.current_stage == 'None'
    # check that we can get model from registry
    versions = registry.get_model_latest_verisons(model_name=model_name)
    assert len(versions) == 1
    assert versions[0].name == model_version.name
    assert versions[0].version == model_version.version
    assert versions[0].current_stage == model_version.current_stage
    # there shouldn't be any production models
    production_version = registry.get_production_model(model_name=model_name)
    assert production_version is None
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run is None
    # transition newly registered model into production, then archived, then back to production
    registered_version = exp.last_run.put_model_in_production(model_name=model_name)
    # if we've already registered the model (above) then we shouldn't re-register, so check
    # that the model version is the same as it was before we put in production
    assert registered_version.version == model_version.version
    # check that the model is in production
    production_version = registry.get_production_model(model_name=model_name)
    assert production_version.run_id == exp.last_run.run_id
    assert production_version.version == model_version.version
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run.run_id == exp.last_run.run_id
    assert production_run.name == exp.last_run.name

    # now transition to archived
    new_version = exp.last_run.set_model_stage(model_name=model_name, to_stage=MLStage.ARCHIVED)
    assert new_version.name == model_version.name
    assert new_version.version == model_version.version
    assert new_version.current_stage == MLStage.ARCHIVED.value
    # now we shouldn't have a model in production
    production_version = registry.get_production_model(model_name=model_name)
    assert production_version is None
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run is None

    # now put back in production
    new_version = exp.last_run.set_model_stage(model_name=model_name, to_stage=MLStage.PRODUCTION)
    assert new_version.name == model_version.name
    assert new_version.version == model_version.version
    assert new_version.current_stage == MLStage.PRODUCTION.value
    # check that the model is in production
    production_version = registry.get_production_model(model_name=model_name)
    assert production_version.run_id == exp.last_run.run_id
    assert production_version.version == model_version.version
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run.run_id == exp.last_run.run_id
    assert production_run.name == exp.last_run.name

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

    exp = registry.get_experiment_by_name(exp_name=experiment_name)
    assert exp.last_run.name == tracker.last_run_name
    assert len(exp.runs) == 2
    assert exp.last_run.exp_name == experiment_name
    assert exp.last_run.exp_id == '1'
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
    assert runs[0].exp_name == experiment_name
    assert runs[0].exp_id == '1'
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
    versions = registry.get_model_latest_verisons(model_name=model_name)
    assert len(versions) == 1
    assert versions[0].version == '1'

    # don't register model directly this time, instead, register by putting into production
    # get the production model, check that the version is still 1
    previous_prod_version = registry.get_production_model(model_name=model_name)
    assert previous_prod_version.name == model_version.name
    assert previous_prod_version.version == '1'
    assert previous_prod_version.current_stage == MLStage.PRODUCTION.value
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run.run_id != exp.last_run.run_id
    assert production_run.name != exp.last_run.name
    assert production_run.run_id == exp.runs[1].run_id
    assert production_run.name == exp.runs[1].name

    registered_version = exp.last_run.put_model_in_production(model_name=model_name)
    new_prod_version = registry.get_production_model(model_name=model_name)
    assert new_prod_version.name == model_version.name
    assert new_prod_version.version == '2'
    assert new_prod_version.current_stage == MLStage.PRODUCTION.value
    production_run = registry.get_production_run(model_name=model_name)
    assert production_run.run_id == exp.last_run.run_id
    assert production_run.name == exp.last_run.name

    # check that previous version is now archived
    previous_model_version = registry.client.get_model_version(
        name=model_name,
        version=previous_prod_version.version
    )
    assert previous_model_version.current_stage == MLStage.ARCHIVED.value
