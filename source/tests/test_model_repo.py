import pytest
import os
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.exceptions
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

from helpsk.sklearn_eval import MLExperimentResults
import source.domain.classification_search_space as css
from source.service.experiment import Tracker, ModelRegistry, Experiment, Run


@pytest.mark.usefixtures('start_ml_server')
def test_services(tracking_uri, data_split):
    experiment_name = 'test_experiment'
    score = 'roc_auc'
    x_train, x_test, y_train, y_test = data_split

    # experiment does not exist at this point
    registry = ModelRegistry(tracking_uri=tracking_uri)
    exp = registry.get_experiment(experiment_name=experiment_name)
    assert exp is None

    tracker = Tracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        tags=dict(type='BayesSearchCV')
    )

    def run_experiment(fake_score, fake_params):
        with tracker:
            tracker.log_text("run 2", "run.txt")
            assert not os.path.isdir(Tracker.TEMP_DIR)
            tracker.log_model(model='mock')
            tracker.log_metric(metric=score, metric_value=fake_score)
            tracker.log_params(best_params=fake_params)
            tracker.log_pickle(obj=x_train, file_name='x_train.pkl')
            tracker.log_pickle(obj=x_test, file_name='x_test.pkl')
            tracker.log_pickle(obj=y_train, file_name='y_train.pkl')
            tracker.log_pickle(obj=y_test, file_name='y_test.pkl')

    run_experiment(fake_score=0.9, fake_params={'param1': 'value', 'param2': 2})
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
    assert exp.last_run.name == tracker.last_run_name

    runs = exp.runs
    assert len(runs) == 1
    assert runs[0].name == tracker.last_run_name
    assert runs[0].experiment_name == experiment_name
    assert runs[0].experiment_id == '1'
    assert runs[0].start_time is not None
    assert runs[0].end_time is not None
    runs[0].mlflow_entity.data
    assert runs[0].metrics == {'roc_auc': 0.9}




    run_experiment(fake_score=0.80, fake_params={'param1': 'value2', 'param2': 3})
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




