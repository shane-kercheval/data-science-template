from __future__ import annotations
import os
from typing import Callable
from collections.abc import Iterable
from enum import Enum, unique
import shutil
from pathlib import Path
from functools import cache
import datetime
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import sklearn.base

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import to_pickle


class TrackerNotRunning(Exception):
    pass


class Tracker:

    TEMP_DIR = '/tmp/0946a999-6cdd-400a-8640-7b5e29788b4c'

    def __init__(
            self,
            experiment_name: str,
            tags: str,
            registry: ModelRegistry):

        self.registry = registry
        self.experiment_name = experiment_name
        self.tags = tags
        self.start_time = None
        self.end_time = None
        self.last_run_name = None
        mlflow.set_tracking_uri(registry.tracking_uri)
        mlflow.set_experiment(experiment_name)

    def __enter__(self):
        """
        Note that mlflow already has a context manager for start_run; the point is not to create
        a context manager; the point is to hide the details of mlflow and allow the user an easy
        interface.
        """
        self.start_time = datetime.datetime.now()
        self.last_run_name = f'{self.start_time:%Y_%m_%d_%H_%M_%S}'
        mlflow.start_run(
            run_name=self.last_run_name,
            description=self.last_run_name,
            tags=self.tags
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = datetime.datetime.now()
        mlflow.end_run()

    @property
    def last_run(self) -> Run:
        return Run.load(
            experiment_name=self.experiment_name,
            run_name=self.last_run_name,
            registry=self.registry
        )

    @property
    def elapsed_seconds(self):
        return (self.end_time - self.start_time).total_seconds()

    def log_model(self, model):
        """
        NOTE: document Wrapped model
        """
        mlflow.sklearn.log_model(
            sk_model=SklearnModelWrapper(model),
            artifact_path='model',
        )

    def log_metric(self, metric, metric_value):
        mlflow.log_metric(metric, metric_value)

    def log_params(self, params):
        params = params.copy()
        _ = params.pop('model', None)
        mlflow.log_params(params=params)

    def log_artifact(self, obj: object, file_name: str, to_file: Callable[[object, str], None]):
        """
        In order to log to MLFlow, it appears the object being logged must be on the file system.

        This is a general helper function that takes an object, saves to to the file system, and
        then logs to MLFlow.

        Args:
            obj: object to log
            file_name: name of the file as it will appear in MLflow
            to_file: function that saves `obj` to the local file system given a path
        """
        if self.last_run_name is None:
            raise TrackerNotRunning()

        try:
            Path(Tracker.TEMP_DIR).mkdir(exist_ok=False)
            file_path = os.path.join(Tracker.TEMP_DIR, file_name)
            to_file(obj, file_path)
            mlflow.log_artifact(local_path=file_path)
        finally:
            shutil.rmtree(Tracker.TEMP_DIR)

    def log_text(self, value: str, file_name: str):
        """
        Log `value` to MLFflow as a text file with `file_name`

        Args:
            value: text value to log to MLFlow
            file_name: name of the file that will appear in MLFlow
        """
        def save_text(value, file_path):
            with open(file_path, 'w') as handle:
                handle.write(value)

        self.log_artifact(obj=value, file_name=file_name, to_file=lambda x, y: save_text(x, y))

    def log_pickle(self, obj: object, file_name: str):
        """
        Log `obj` to MLFflow as a pickled file with `file_name`

        Args:
            obj: object to pickle and log to MLFlow
            file_name: name of the file that will appear in MLFlow
        """
        self.log_artifact(obj=obj, file_name=file_name, to_file=lambda x, y: to_pickle(x, y))

    def log_ml_results(self, results: MLExperimentResults, file_name: str):
        """
        Log `results` to MLFflow as a YAML file with `file_name`

        Args:
            obj: results save and log to MLFlow
            file_name: name of the YAML file that will appear in MLFlow
        """
        self.log_artifact(
            obj=results,
            file_name=file_name,
            to_file=lambda x, y: x.to_yaml_file(yaml_file_name=y)
        )


@unique
class MLStage(Enum):
    STAGING = 'Staging'
    PRODUCTION = 'Production'
    ARCHIVED = 'Archived'


class ModelRegistry:
    """
    NOTE that these entities (e.g. Experiment/Run) are not only coupled to Model Registry they're
    actaually coupled to MLFlow because they the underlying MLFlow objects. Ideally, but overkill,
    would be to use an ORM & Repository pattern to truly decouple entitles from MLFlow. But the
    goal here is to simply have layers of abstraction, not to decouple, but to make the code easier
    to read, use, test, etc.
    """
    def __init__(self, tracking_uri):
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri

    def clear_cache(self):
        self.get_experiment.cache_clear()
        self.get_runs.cache_clear()
        self.get_runs_from_id.cache_clear()
        self.get_run.cache_clear()
        self.download_artifact.cache_clear()
        self.get_model_latest_verisons.cache_clear()
        self.get_production_model.cache_clear()

    def track_experiment(self, experiment_name: str, tags: str) -> Tracker:
        return Tracker(
            experiment_name=experiment_name,
            tags=tags,
            registry=self)

    @cache
    def get_experiment(self, experiment_name: str) -> mlflow.entities.experiment.Experiment:
        return self.client.get_experiment_by_name(experiment_name)

    @cache
    def get_runs(self, experiment_name: str) -> list[mlflow.entities.run.Run]:
        exp = self.get_experiment(experiment_name)
        return list(self.get_runs_from_id(experiment_id=exp.experiment_id))

    @cache
    def get_runs_from_id(self, experiment_id: str) -> list[mlflow.entities.run.Run]:
        return list(self.client.search_runs([experiment_id]))

    @cache
    def get_run(self, experiment_name: str, run_name: str) -> mlflow.entities.run.Run:
        return next(
            x for x in self.get_runs(experiment_name)
            if x.data.tags['mlflow.runName'] == run_name
        )

    @cache
    def download_artifact(
            self,
            run_id,
            artifact_name: str,
            read_from: Callable[[object, str], None]):
        return read_from(self.client.download_artifacts(run_id=run_id, path=artifact_name))

    def register_model(self, run_id: str, model_name: str) -> ModelVersion:
        model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        self.clear_cache()
        return model_version

    @cache
    def get_model_latest_verisons(
            self,
            model_name: str,
            stages: tuple[MLStage] | None = None) -> list[ModelVersion]:
        """
        Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.
        """
        if stages is not None:
            if not isinstance(stages, Iterable):
                stages = [stages]
            stages = [x.value for x in stages]
        try:
            versions = self.client.get_latest_versions(
                name=model_name,
                stages=stages,
            )
        except mlflow.exceptions.RestException:
            versions = []

        return versions

    @cache
    def get_production_model(self, model_name: str) -> ModelVersion | None:
        versions = self.get_model_latest_verisons(
            model_name=model_name,
            stages=(MLStage.PRODUCTION,)
        )
        if len(versions) == 0:
            return None

        assert len(versions) == 1
        return versions[0]

    def transition_model_to_stage(
            self,
            model_name: str,
            model_version: str,
            to_stage: MLStage) -> ModelVersion:
        model_version = self.client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=to_stage.value,
        )
        self.clear_cache()
        return model_version

    # def transition_last_model(model_name: str, stage: str) -> ModelVersion:
    #     """
    #     Register the model associated with the last active run and transition the stage to `stage`.

    #     args:
    #         ml_client: MlflowClient object
    #         model_name: name of the registered model
    #         stage: stage to transition model in last active run. (Staging|Archived|Production|None)
    #     """
    #     model_version = mlflow.register_model(
    #         model_uri=f"runs:/{mlflow.last_active_run().info.run_id}/model",
    #         name=model_name
    #     )
    #     _ = ml_client.transition_model_version_stage(
    #         name=model_name,
    #         version=str(model_version.version),
    #         stage=stage,
    #     )
    #     return model_version


class MLFlowEntity:
    def __init__(self, entity, registry: ModelRegistry):
        self.mlflow_entity = entity
        self.mlflow_registry = registry

    def __getattr__(self, name):
        """Expose the underlying attributes of the MLFlow entity"""
        return getattr(self.mlflow_entity, name)

    def clear_cache(self):
        self.mlflow_registry.clear_cache()


class Run(MLFlowEntity):
    def __init__(self, experiment_name, entity, registry: ModelRegistry):
        super().__init__(entity=entity, registry=registry)
        self.experiment_name = experiment_name

    @classmethod
    def load(cls, experiment_name: str, run_name: str, registry: ModelRegistry):
        entity = registry.get_run(
            experiment_name=experiment_name,
            run_name=run_name,
        )
        assert entity.data.tags['mlflow.runName'] == run_name
        return cls(experiment_name=experiment_name, entity=entity, registry=registry)

    @classmethod
    def load_from_id(cls, experiment_name, run_id: str, registry: ModelRegistry):
        entity = registry.client.get_run(run_id=run_id)
        assert entity.info.run_id == run_id
        return cls(experiment_name=experiment_name, entity=entity, registry=registry)

    @property
    def name(self) -> str:
        return self.mlflow_entity.data.tags['mlflow.runName']

    @property
    def id(self):
        return self.mlflow_entity.info.run_id

    @property
    def experiment_id(self):
        return self.mlflow_entity.info.experiment_id

    @property
    def start_time(self):
        return datetime.datetime.fromtimestamp(self.mlflow_entity.info.start_time/1000.0)

    @property
    def end_time(self):
        return datetime.datetime.fromtimestamp(self.mlflow_entity.info.end_time/1000.0)

    @property
    def metrics(self):
        return self.mlflow_entity.data.metrics

    @property
    def params(self):
        return self.mlflow_entity.data.params

    @property
    def tags(self):
        return self.mlflow_entity.data.tags

    def download_artifact(self, artifact_name: str, read_from: Callable[[object, str], None]):
        return self.mlflow_registry.download_artifact(
            run_id=self.id,
            artifact_name=artifact_name,
            read_from=read_from
        )

    def register_model(self, model_name: str) -> ModelVersion:
        return self.mlflow_registry.register_model(run_id=self.id, model_name=model_name)

    # def set_model_stage(self, model_name, to_stage: str) -> ModelVersion:
    #     model_version = self.mlflow_registry.get_model_latest_verisons(model_name=model_name)
    #     return self.mlflow_registry.transition_model_to_stage(
    #         model_name=model_name,
    #         model_version=str(model_version.version),
    #         to_stage=to_stage,
    #     )


class Experiment(MLFlowEntity):
    def __init__(self, entity, registry: ModelRegistry):
        super().__init__(entity=entity, registry=registry)

    @classmethod
    def load(cls, experiment_name: str, registry: ModelRegistry):
        entity = registry.get_experiment(experiment_name=experiment_name)
        if entity is None:
            return None
        assert entity.name == experiment_name
        return cls(entity=entity, registry=registry)

    @property
    def id(self):
        return self.mlflow_entity.experiment_id

    @property
    def runs(self) -> list[Run]:
        return [
            Run.load(
                experiment_name=self.name,
                run_name=x.data.tags['mlflow.runName'],
                registry=self.mlflow_registry
            )
            for x in self.mlflow_registry.get_runs_from_id(experiment_id=self.experiment_id)
        ]

    @property
    def last_run(self) -> Run:
        sorted_runs = sorted(self.runs, key=lambda x: x.start_time, reverse=True)
        if len(sorted_runs) == 0:
            return None

        return sorted_runs[0]


class SklearnModelWrapper(sklearn.base.BaseEstimator):
    """
    The predict method of various sklearn models returns a binary classification (0 or 1).
    The following code creates a wrapper function, SklearnModelWrapper, that uses
    the predict_proba method to return the probability that the observation belongs to each class.
    Code from:
    https://docs.azure.cn/en-us/databricks/_static/notebooks/mlflow/mlflow-end-to-end-example-azure.html
    """
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        return self.predict_proba(data=data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)[:, 1]
