import os
import shutil
from pathlib import Path
import mlflow
from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import to_pickle


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
