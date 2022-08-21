import pytest
import os
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions
import pandas as pd

from helpsk.validation import dataframes_match
from source.service.experiment import Tracker, ModelRegistry, Experiment, Run


@pytest.mark.usefixtures('start_ml_server')
def test_services(tracking_uri, data_split):
    experiment_name = 'test_experiment'
    metric = 'roc_auc'
    x_train, x_test, y_train, y_test = data_split

    # experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment(experiment_name=experiment_name)
    assert exp is None

    def read_text(file_path):
        with open(file_path, 'r') as handle:
            return handle.read()

    tracker = Tracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        tags=dict(type='BayesSearchCV')
    )

    def run_experiment(fake_metric: str, fake_params: dict, fake_value: str, fake_name: str):
        with tracker:
            tracker.log_text(fake_value, fake_name)
            assert not os.path.isdir(Tracker.TEMP_DIR)
            tracker.log_model(model='mock')
            tracker.log_metric(metric=metric, metric_value=fake_metric)
            tracker.log_params(best_params=fake_params)
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
    assert exp.last_run.start_time == max(x.start_time for x in runs)
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
    # Second Experiment
    ####
    last_exp_seconds = tracker.elapsed_seconds
    last_exp_run_name = tracker.last_run_name
    last_exp_start_time = exp.last_run.start_time
    last_exp_end_time = exp.last_run.end_time

    run_experiment(
        fake_metric=0.95,
        fake_params={'param2': 'value3', 'param3': '4'},
        fake_value='run 3',
        fake_name='run 3 text'
    )

    assert tracker.elapsed_seconds > 0
    assert tracker.elapsed_seconds != last_exp_seconds
    assert tracker.last_run_name is not None
    assert tracker.last_run_name != last_exp_run_name

    # same experiment
    exp = Experiment.load(experiment_name=experiment_name, registry=registry)
    assert exp is not None
    assert exp.name == experiment_name
    assert exp.id == '1'

    # cache isn't cleared
    assert exp.last_run.name == last_exp_run_name
    assert len(exp.runs) == 2

    registry.clear_cache()
    assert exp.last_run.name == tracker.last_run_name
    assert len(exp.runs) == 3
    assert exp.last_run.experiment_name == experiment_name
    assert exp.last_run.experiment_id == '1'
    assert exp.last_run.start_time is not None
    assert exp.last_run.start_time > last_exp_start_time
    assert exp.last_run.start_time == max(x.start_time for x in runs)
    assert exp.last_run.end_time is not None
    assert exp.last_run.end_time > last_exp_end_time
    assert exp.last_run.metrics == {'roc_auc': 0.95}
    assert exp.last_run.params == {'param2': 'value3', 'param3': '4'}
    assert exp.last_run.tags['type'] == 'BayesSearchCV'
    logged_value = exp.last_run.download_artifact(artifact_name='run 3 text', read_from=read_text)
    assert logged_value == 'run 3'
    downloaded_x_train = exp.last_run.download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])

    runs = exp.runs
    assert len(runs) == 3
    assert runs[0].name == tracker.last_run_name
    assert runs[0].experiment_name == experiment_name
    assert runs[0].experiment_id == '1'
    assert runs[0].start_time is not None
    assert runs[0].end_time is not None
    assert runs[0].metrics == {'roc_auc': 0.95}
    assert runs[0].params == {'param1': 'value2', 'param2': '3'}
    assert runs[0].tags['type'] == 'BayesSearchCV'
    logged_value = runs[0].download_artifact(artifact_name='run 2 text', read_from=read_text)
    assert logged_value == 'run 2'
    downloaded_x_train = runs[0].download_artifact('x_train.pkl', read_from=pd.read_pickle)
    assert dataframes_match([downloaded_x_train, x_train])





def temp():
    run_experiment(fake_metric=0.80, fake_params={'param1': 'value2', 'param2': 3})
    assert tracker.last_run_name is not None
    exp = registry.get_experiment('test_experiment')
    registry.clear_cache()
    exp = registry.get_experiment('test_experiment')


    tracker.last_run_name

    
    exp = registry.get_experiment('test_experiment')
    assert exp.name == 'test_experiment'
    assert exp.experiment_id == '1'
    runs = registry.get_runs(experiment_name='test_experiment')
    type(runs)
    runs_from_id = list(registry.get_runs_from_id('1'))

    run_names = [x.data.tags['mlflow.runName'] for x in runs_from_id]
    
    
    datetime.datetime.fromtimestamp(run.mlflow_entity.info.start_time/1000.0)


    registry.get_run(experiment_name='test_experiment', run_name=tracker.last_run_name)

    def read_txt(path):
        with open(path, 'r') as handle:
            value = handle.read()
        return value

    client.download_artifacts(run_id=run.id, path='test_2.txt')
    read_txt('/tmp/pytest-of-root/pytest-30/ml_server0/mlflow-artifact-root/1/7d5d0c94087145fd9b1b67383d496cbc/artifacts/test_2.txt')
    read_txt('Dockerfile')
    read

    run = Run.load(experiment_name='test_experiment', run_name=tracker.last_run_name, registry=registry)

    run.mlflow_entity
    run.experiment_name
    run.name
    run.id
    run.start_time
    run.start_time
    run.mlflow_entity.info.start_time
    run.mlflow_entity.info.experiment_id
    run.mlflow_entity.info.start_time
    run.id

    value = registry.get_artifact(run_id=run.id, artifact_name='test_2.txt', read_from=read_txt)
    assert value == 'Hello'





    import datetime
    datetime.datetime(run.mlflow_entity.info.start_time)
    DateFormat = '%d%m%H%M%S'
    import dateutil
    dateutil.parser.parse(run.mlflow_entity.info.start_time).strftime(DateFormat)

    run.mlflow_entity.info.end_time

    exp = Experiment.load(experiment_name='test_experiment', registry=registry)
    assert exp.mlflow_entity is not None
    assert exp.name == 'test_experiment'
    assert exp.id == '1'
    assert exp.lifecycle_stage == 'active'


    assert exp.last_run.start_time == max(x.start_time for x in runs)

    runs[0].start_time
    sorted(runs, key=lambda x: x.start_time, reverse=True)
    sorted(runs, key=lambda x: x.start_time)
    exp.get

    

    exp.mlflow_registry.get_runs.cache_clear()
    exp.mlflow_registry.get_runs_from_id.cache_clear()
    exp.mlflow_registry.get_run.cache_clear()
    exp.mlflow_registry.get_experiment.cache_clear()

    
    exp.mlflow_registry.__dict__

    exp.__dict__





    client = MlflowClient(tracking_uri="http://0.0.0.0:1235")
    experiment_data = client.get_experiment_by_name(experiment_name)
    experiment_data
    experiment_data['experiment_id']

    mlflow.search_runs(["1"])
    temp = client.search_runs(["1"])
    type(list(temp)[0])

    

    tracker.elapsed_seconds
    tracker.experiment_name
    tracker.tags
    tracker.tracking_uri


    temp = DynamicAttributes(arg1='a')
    temp.arg1


    last_model = exp_registry.get_most_recent_model()
    last_model = exp_registry.transition_model()
    production_model = exp_registry.get_production_model()




