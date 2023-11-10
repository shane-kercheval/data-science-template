"""Wrappers for MLFlow entities (e.g. Experiment, Run, ModelVersion)."""

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
import pandas as pd
import sklearn.base

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import to_pickle


class TrackerNotRunningError(Exception):
    """Exception raised when trying to log to MLFlow when the tracker is not running."""


class Tracker:
    """
    Encapsulates the logic for starting the run in mlflow. It's meant to be used as a
    context manager, and it also starts and stops a timer to track the duration of an mlrun.
    """

    TEMP_DIR = '/tmp/0946a999-6cdd-400a-8640-7b5e29788b4c'

    def __init__(
            self,
            exp_name: str,
            tags: str,
            registry: ModelRegistry):

        self.registry = registry
        self.exp_name = exp_name
        self.tags = tags
        self.start_time = None
        self.end_time = None
        self.last_run_name = None
        self._last_run = None
        mlflow.set_tracking_uri(registry.tracking_uri)
        mlflow.set_experiment(exp_name)

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
            tags=self.tags,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        self.end_time = datetime.datetime.now()
        mlflow.end_run()
        self.registry.clear_cache()

    @property
    def last_run(self) -> Run:
        """Returns the last run associated with this tracker."""
        if self._last_run is None:
            self._last_run = self.registry.get_run_by_name(
                exp_name=self.exp_name,
                run_name=self.last_run_name,
            )
        return self._last_run

    @property
    def elapsed_seconds(self) -> float:
        """Returns the number of seconds between start and end time of the tracker."""
        return (self.end_time - self.start_time).total_seconds()

    def log_model(self, model: sklearn.base.BaseEstimator) -> None:
        """Log the model to MLFlow."""
        mlflow.sklearn.log_model(
            sk_model=SklearnModelWrapper(model),
            artifact_path='model',
        )

    def log_metric(self, metric: str, metric_value: float) -> None:
        """Log the metric to MLFlow."""
        mlflow.log_metric(metric, metric_value)

    def log_params(self, params: dict) -> None:
        """Log the params to MLFlow."""
        params = params.copy()
        _ = params.pop('model', None)
        mlflow.log_params(params=params)

    def log_artifact(self,
            obj: object,
            file_name: str,
            to_file: Callable[[object,
            str],
            None]) -> None:
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
            raise TrackerNotRunningError()

        try:
            Path(Tracker.TEMP_DIR).mkdir(exist_ok=False)
            file_path = os.path.join(Tracker.TEMP_DIR, file_name)
            to_file(obj, file_path)
            mlflow.log_artifact(local_path=file_path)
        finally:
            shutil.rmtree(Tracker.TEMP_DIR)

    def log_text(self, value: str, file_name: str) -> None:
        """
        Log `value` to MLFflow as a text file with `file_name`.

        Args:
            value: text value to log to MLFlow
            file_name: name of the file that will appear in MLFlow
        """
        def save_text(value: str, file_path: str):  # noqa: ANN202
            with open(file_path, 'w') as handle:
                handle.write(value)

        self.log_artifact(obj=value, file_name=file_name, to_file=lambda x, y: save_text(x, y))

    def log_pickle(self, obj: object, file_name: str) -> None:
        """
        Log `obj` to MLFflow as a pickled file with `file_name`.

        Args:
            obj: object to pickle and log to MLFlow
            file_name: name of the file that will appear in MLFlow
        """
        self.log_artifact(obj=obj, file_name=file_name, to_file=lambda x, y: to_pickle(x, y))

    def log_ml_results(self, results: MLExperimentResults, file_name: str) -> None:
        """
        Log `results` to MLFflow as a YAML file with `file_name`.

        Args:
            results: results save and log to MLFlow
            file_name: name of the YAML file that will appear in MLFlow
        """
        self.log_artifact(
            obj=results,
            file_name=file_name,
            to_file=lambda x, y: x.to_yaml_file(yaml_file_name=y),
        )


@unique
class MLStage(Enum):
    """The stages of a model in the model registry."""

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

    def __init__(self, tracking_uri: str):
        """
        A ModelRegistry object.

        Args:
            tracking_uri: the URI for the underlying experimentation client (e.g. mlflow)
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.tracking_uri = tracking_uri

    def clear_cache(self) -> None:
        """Clear the cache for all the cached methods."""
        self._get_experiment_by_name.cache_clear()
        self._get_experiment_by_id.cache_clear()
        self._get_runs_by_experiment_name.cache_clear()
        self._get_runs_by_experiment_id.cache_clear()
        self._get_run_by_name.cache_clear()
        self._get_run_by_id.cache_clear()
        self.download_artifact.cache_clear()
        self.get_model_latest_verisons.cache_clear()
        self.get_production_model.cache_clear()
        self.get_production_run.cache_clear()

    def track_experiment(self, exp_name: str, tags: str) -> Tracker:
        """Start an MLFlow experiment and return a Tracker object."""
        return Tracker(exp_name=exp_name, tags=tags, registry=self)

    @cache
    def _get_experiment_by_name(self, exp_name: str) -> mlflow.entities.experiment.Experiment:  # noqa
        """
        Return MLFlow Experiment based on experiment name.

        Args:
            name: the experiment name
        """
        return self.client.get_experiment_by_name(name=exp_name)

    @cache
    def _get_experiment_by_id(self, exp_id: str) -> mlflow.entities.experiment.Experiment:
        """
        Return MLFlow Experiment based on experiment ID.

        Args:
            exp_id: the experiment id
        """
        try:
            return self.client.get_experiment(experiment_id=exp_id)
        except mlflow.exceptions.RestException:
            return None

    @cache
    def _get_runs_by_experiment_name(self, exp_name: str) -> list[mlflow.entities.run.Run]:  # noqa
        """
        Return list of MLFlow Run objects based on experiment name.

        Args:
            name: the experiment name
        """
        exp = self._get_experiment_by_name(exp_name=exp_name)
        if exp is None:
            return None
        return list(self._get_runs_by_experiment_id(exp_id=exp.experiment_id))

    @cache
    def _get_runs_by_experiment_id(self, exp_id: str) -> list[mlflow.entities.run.Run]:
        """
        Return list of MLFlow Run objects based on experiment name.

        Args:
            exp_id: the experiment id
        """
        return list(self.client.search_runs([exp_id]))

    @cache
    def _get_run_by_name(self, exp_name: str, run_name: str) -> mlflow.entities.run.Run:
        """
        Return MLFlow Run object based on run name (requires experiment name).

        Args:
            exp_name: the name of the experiment
            run_name: the name of the run
        """
        runs = self._get_runs_by_experiment_name(exp_name=exp_name)
        if runs is None:
            return None
        return next(x for x in runs if x.data.tags['mlflow.runName'] == run_name)

    @cache
    def _get_run_by_id(self, run_id: str) -> mlflow.entities.run.Run:
        """
        Return MLFlow Run object based on run id.

        Args:
            run_id: the id of the run
        """
        try:
            return self.client.get_run(run_id=run_id)
        except mlflow.exceptions.MlflowException:
            return None

    def get_experiment_by_name(self, exp_name: str) -> Experiment:
        """Queryies MLFlow and returns the experiment based on the name name."""
        experiment = self._get_experiment_by_name(exp_name=exp_name)
        if experiment is None:
            return None
        return Experiment(entity=experiment, registry=self)

    def get_experiment_by_id(self, exp_id: str) -> Experiment:
        """Queryies MLFlow and returns the experiment based on the name id."""
        experiment = self._get_experiment_by_id(exp_id=exp_id)
        if experiment is None:
            return None
        return Experiment(entity=experiment, registry=self)

    def get_run_by_name(self, exp_name: str, run_name: str) -> Run:
        """Queryies MLFlow and returns the run based on the experiment and run name."""
        run = self._get_run_by_name(exp_name=exp_name, run_name=run_name)
        if run is None:
            return None
        return Run(entity=run, registry=self)

    def get_run_by_id(self, run_id: str) -> Run:
        """Queryies MLFlow and returns the run based on the run id."""
        run = self._get_run_by_id(run_id=run_id)
        if run is None:
            return None
        return Run(entity=run, registry=self)

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
        """Returns the model that is currently in production."""
        versions = self.get_model_latest_verisons(
            model_name=model_name,
            stages=(MLStage.PRODUCTION),
        )
        if len(versions) == 0:
            return None

        assert len(versions) == 1
        return versions[0]

    @cache
    def get_production_run(self, model_name: str) -> Run:
        """Returns the run associated with the model that is currently in production."""
        # get model associated with production
        # get the run_id associated with that model
        production_model = self.get_production_model(model_name=model_name)
        if production_model is None:
            return None
        return self.get_run_by_id(
            run_id=production_model.run_id,
        )

    @cache
    def download_artifact(
            self,
            run_id: str,
            artifact_name: str,
            read_from: Callable[[object, str], None]) -> object | None:
        """Download the artifact from the run and read it into memory."""
        try:
            path = self.client.download_artifacts(run_id=run_id, path=artifact_name)
            return read_from(path)
        except mlflow.exceptions.RestException:
            return None

    def register_model(self, run_id: str, model_name: str) -> ModelVersion:
        """Register the model associated with the run id with the model registry."""
        model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)
        self.clear_cache()
        return model_version

    def transition_model_to_stage(
            self,
            model_name: str,
            model_version: str,
            to_stage: MLStage) -> ModelVersion:
        """Transition the model to the requested stage."""
        model_version = self.client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=to_stage.value,
        )
        self.clear_cache()
        return model_version

    def put_model_in_production(
            self,
            model_name: str,
            model_version: ModelVersion) -> ModelVersion:
        """
        Put the model into production. If there is already a model in production, archive that
        model first.
        """
        # get model that is currently in production
        # if model is in production, move to archive
        # put new model into production
        production_model = self.get_production_model(model_name=model_name)
        if production_model is not None:
            _ = self.transition_model_to_stage(
                model_name=production_model.name,
                model_version=production_model.version,
                to_stage=MLStage.ARCHIVED,
            )
        return self.transition_model_to_stage(
            model_name=model_name,
            model_version=model_version,
            to_stage=MLStage.PRODUCTION,
        )


class MLFlowEntity:
    """Base class for MLFlow entities (e.g. Experiment, Run, ModelVersion)."""

    def __init__(self, entity, registry: ModelRegistry):  # noqa: ANN001
        self.mlflow_entity = entity
        self.mlflow_registry = registry

    def __getattr__(self, name: str):
        """Expose the underlying attributes of the MLFlow entity."""
        return getattr(self.mlflow_entity, name)

    def clear_cache(self) -> None:
        """Clear the cache for all the cached methods."""
        self.mlflow_registry.clear_cache()


class Run(MLFlowEntity):
    """
    A wrapper for the MLFlow Run object. This class exposes the underlying MLFlow Run object
    attributes and methods, but also adds additional properties and methods to make it easier to
    work with the MLFlow Run object.
    """

    def __init__(self, entity: str, registry: ModelRegistry):
        super().__init__(entity=entity, registry=registry)
        self.model_version = None  # gets set when model is registered

    @property
    def name(self) -> str:
        """Returns the name of the run."""
        return self.mlflow_entity.data.tags['mlflow.runName']

    @property
    def run_id(self) -> str:
        """Returns the run id."""
        return self.mlflow_entity.info.run_id

    @property
    def exp_id(self) -> str:
        """Returns the experiment id."""
        return self.mlflow_entity.info.experiment_id

    @property
    def exp_name(self) -> str:
        """Returns the experiment name."""
        return self.mlflow_registry._get_experiment_by_id(self.exp_id).name

    @property
    def start_time(self) -> datetime.datetime:
        """Returns the start time of the run."""
        return datetime.datetime.fromtimestamp(self.mlflow_entity.info.start_time/1000.0)

    @property
    def end_time(self) -> datetime.datetime:
        """Returns the end time of the run."""
        return datetime.datetime.fromtimestamp(self.mlflow_entity.info.end_time/1000.0)

    @property
    def metrics(self) -> dict[str, float]:
        """Returns the metrics associated with the run."""
        return self.mlflow_entity.data.metrics

    @property
    def params(self) -> dict[str, str]:
        """Returns the params associated with the run."""
        return self.mlflow_entity.data.params

    @property
    def tags(self) -> dict[str, str]:
        """Returns the tags associated with the run."""
        return self.mlflow_entity.data.tags

    def download_artifact(self,
        artifact_name: str,
        read_from: Callable[[object, str], None]) -> object | None:
        """Download the artifact from the run and read it into memory."""
        return self.mlflow_registry.download_artifact(
            run_id=self.run_id,
            artifact_name=artifact_name,
            read_from=read_from,
        )

    def register_model(self, model_name: str) -> ModelVersion:
        """
        Only registers the model if the model has not already been registered (by this locally
        created object; i.e. doesn't check MLFlow, just caches the result locally when
        registering).
        """
        if self.model_version is None:
            self.model_version = self.mlflow_registry.register_model(
                run_id=self.run_id,
                model_name=model_name,
            )
        return self.model_version

    def put_model_in_production(self, model_name: str) -> ModelVersion:
        """
        If there is currently a model in production, archive that model, then transition the
        model associated with this run into production.
        """
        model_verison = self.register_model(model_name=model_name)
        # this update will affect the underlying ModelVersion object; need to update the
        # corresponding instance object
        self.model_version = self.mlflow_registry.put_model_in_production(
            model_name=model_name,
            model_version=model_verison.version,
        )
        return self.model_version

    def set_model_stage(self, model_name: str, to_stage: MLStage) -> ModelVersion:
        """Does not transition current models in production out of production."""
        model_version = self.register_model(model_name=model_name)
        # this update will affect the underlying ModelVersion object; need to update the
        # corresponding instance object
        self.model_version = self.mlflow_registry.transition_model_to_stage(
            model_name=model_name,
            model_version=model_version.version,
            to_stage=to_stage,
        )
        return self.model_version


class Experiment(MLFlowEntity):
    """
    A wrapper for the MLFlow Experiment object. This class exposes the underlying MLFlow Experiment
    object attributes and methods, but also adds additional properties and methods to make it
    easier to work with the MLFlow Experiment object.
    """

    def __init__(self, entity, registry: ModelRegistry):  # noqa: ANN001
        super().__init__(entity=entity, registry=registry)
        self.exp_runs = None

    @property
    def exp_id(self) -> str:
        """Returns the experiment id."""
        return self.mlflow_entity.experiment_id

    @property
    def runs(self) -> list[Run]:
        """Returns the runs associated with the experiment."""
        if self.exp_runs is None:
            exp_runs = self.mlflow_registry._get_runs_by_experiment_id(exp_id=self.exp_id)
            self.exp_runs = [
                self.mlflow_registry.get_run_by_name(
                    exp_name=self.name,
                    run_name=x.data.tags['mlflow.runName'],
                )
                for x in exp_runs
            ]
        return self.exp_runs

    @property
    def last_run(self) -> Run:
        """Returns the last run associated with the experiment."""
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
    https://docs.azure.cn/en-us/databricks/_static/notebooks/mlflow/mlflow-end-to-end-example-azure.html.
    """

    def __init__(self, model: sklearn.base.BaseEstimator):
        self.model = model

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Overrides the default predict method and returns the probability that the observation
        belongs to each class.
        """
        return self.predict_proba(data=data)

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Returns the probability that the observation belongs to each class."""
        return self.model.predict_proba(data)[:, 1]
