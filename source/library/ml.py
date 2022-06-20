import os
import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import to_pickle


def initialize_mlflow(tracking_uri: str, experiment_name: str):
    """Initialize MLFlow"""
    # set up MLFlow
    # mlflow.sklearn.autolog(registered_model_name=registered_model_name)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_pickle(obj: object, file_name: str):
    """
    This method provides a convenient way to log to mlflow without having to manually save the file to local
    storage.
    """
    temp_dir = '0946a999-6cdd-400a-8640-7b5e29788b4c'
    try:
        Path(temp_dir).mkdir(exist_ok=False)
        file_path = os.path.join(temp_dir, file_name)
        to_pickle(obj=obj, path=file_path)
        mlflow.log_artifact(local_path=file_path)
    finally:
        shutil.rmtree(temp_dir)


def log_ml_results(results: MLExperimentResults, file_name: str):
    """
    This method provides a convenient way to log to mlflow without having to manually save the file to local
    storage.
    """
    temp_dir = '0946a999-6cdd-400a-8640-7b5e29788b4c'
    Path(temp_dir).mkdir(exist_ok=False)
    try:
        file_path = os.path.join(temp_dir, file_name)
        results.to_yaml_file(yaml_file_name=file_path)
        mlflow.log_artifact(local_path=file_path)
    finally:
        shutil.rmtree(temp_dir)


def transition_last_model(ml_client: MlflowClient, model_name: str, stage: str) -> ModelVersion:
    """
    Register the model associated with the last active run and transition the stage to `stage`.

    args:
        ml_client: MlflowClient object
        model_name: name of the registered model
        stage: stage to transition model in last active run. (Staging|Archived|Production|None)
    """
    model_version = mlflow.register_model(
        model_uri=f"runs:/{mlflow.last_active_run().info.run_id}/model",
        name=model_name
    )
    _ = ml_client.transition_model_version_stage(
        name=model_name,
        version=str(model_version.version),
        stage=stage,
    )
    return model_version
