"""
This file contains the logic for running ML experiments using BayesSearchCV.
"""
import datetime
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient
from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import read_pickle
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skopt import BayesSearchCV  # noqa

sys.path.append(os.getcwd())
from source.library.utilities import Timer, log_info, log_func  # noqa
import source.library.ml as ml  # noqa
import source.library.classification_search_space as css  # noqa


def run(input_directory: str,
        output_directory: str,
        n_iterations: int,
        n_splits: int,
        n_repeats: int,
        score: str,
        tracking_uri: str) -> str:
    """
    Logic For Running Experiments. The timestamp of the experiment is returned by the
    function. The experiment results are saved to a sub-folder called `experiments__[timestamp]`.

    This function takes the full credit dataset, creates features, and separates the dataset into training
    and test sets. It saves those datasets in `output_directory_data` for future reference.
    The training set is used by BayesSearchCV to search for the best model and the test set is used to
    evaluate the performance of the best model found. The results are saved in the output directory.

    Args:
        input_directory:
            the directory to find credit.pkl
        output_directory:
            the directory to save the results; the results will be saved ot a sub-folder within this directory
            with the timestamp of the experiment
        n_iterations:
            the number of iterations for BayesSearchCV per model (i.e. the number of hyper-parameter
            combinations to search, per model.
        n_splits:
            The number of cross validation splits (e.g. 10-split 2-repeat cross validation).
        n_repeats:
            The number of cross validation repeats (e.g. 10-split 2-repeat cross validation).
        score:
            A string identifying the score to evaluate model performance, e.g. `roc_auc`, `average_precision`,
        tracking_uri:
            MLFlow tracking URI e.g. http://localhost:1234
            Should match host/port in makefile `mlflow_server` command.
    """
    log_func("experiments.run", params=dict(
        input_directory=input_directory,
        output_directory=output_directory,
        n_iterations=n_iterations,
        n_splits=n_splits,
        n_repeats=n_repeats,
        score=score,
    ))
    timestamp = f'{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}'
    log_info(f"Saving data and results to `{output_directory}`")
    log_info("Splitting training & test datasets")
    credit_data = read_pickle(os.path.join(input_directory, 'credit.pkl'))

    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')

    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("credit")
    mlflow.sklearn.autolog(registered_model_name='credit_model')
    with Timer("Running Model Experiments (BayesSearchCV)"):
        with mlflow.start_run(run_name=timestamp,
                              description=timestamp,
                              tags=dict(type='BayesSearchCV',
                                        timestamp=timestamp)):

            bayes_search = BayesSearchCV(
                estimator=css.create_pipeline(data=x_train),
                search_spaces=css.create_search_space(iterations=n_iterations),
                cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42),
                scoring=score,
                refit=True,
                return_train_score=False,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )
            bayes_search.fit(x_train, y_train)
            log_info(f"Best Score: {bayes_search.best_score_}")
            log_info(f"Best Params: {bayes_search.best_params_}")

            results = MLExperimentResults.from_sklearn_search_cv(
                searcher=bayes_search,
                higher_score_is_better=True,
                description='BayesSearchCV',
                parameter_name_mappings=css.get_search_space_mappings(),
            )
            assert bayes_search.scoring == score
            mlflow.log_metric(score, bayes_search.best_score_)
            # mlflow.log_params(bayes_search.best_params_)

            file_name = 'experiment'
            # yaml_file_name = os.path.join(output_directory, f'{file_name}.yaml')
            # model_file_name = os.path.join(output_directory, f'{file_name}_best_estimator.pkl')
            yaml_file_name = f'{file_name}.yaml'
            model_file_name = f'{file_name}_best_estimator.pkl'

            # log_info(f"Saving results of BayesSearchCV to: `{yaml_file_name}`")
            # results.to_yaml_file(yaml_file_name=yaml_file_name)
            # mlflow.log_artifact(local_path=yaml_file_name)
            ml.log_ml_results(results=results, file_name=yaml_file_name)
            ml.log_pickle(obj=bayes_search.best_estimator_, file_name=model_file_name)
            ml.log_pickle(obj=x_train, file_name='x_train.pkl')
            ml.log_pickle(obj=x_test, file_name='x_test.pkl')
            ml.log_pickle(obj=y_train, file_name='y_train.pkl')
            ml.log_pickle(obj=y_test, file_name='y_test.pkl')

    def transition_latest_model_to_production(ml_client: MlflowClient):
        """Get the latest version of credit_model and transitino stage to Production."""
        credit_model = ml_client.get_registered_model(name='credit_model')
        latest_version = max([int(x.version) for x in credit_model.latest_versions])
        log_info(f"Transitioning latest `credit_model` (version {latest_version}) to Production")
        _ = ml_client.transition_model_version_stage(
            name=production_model.name,
            version=str(latest_version),
            stage='Production'
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    production_model = client.get_latest_versions(name='credit_model', stages=['Production'])
    if len(production_model) == 0:
        # put current model into production
        log_info("No models currently in production.")
        transition_latest_model_to_production(ml_client=client)
    else:
        # put latest model into production after archiving model currently in production
        assert len(production_model) == 1  # can only have one model in production
        production_model = production_model[0]
        production_roc_auc = client.get_run(run_id=production_model.run_id).data.metrics['roc_auc']
        if bayes_search.best_score_ > production_roc_auc * 1.025:
            log_info(f"Archiving previous model "
                     f"(version {production_model.version}; roc_auc {production_roc_auc}) and placing new "
                     f"model into Production (roc_auc {bayes_search.best_score_})")
            _ = client.transition_model_version_stage(
                name=production_model.name,
                version=production_model.version,
                stage='Archived'
            )
            transition_latest_model_to_production(ml_client=client)

    return timestamp


# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 
# client.download_artifacts(run_id='d63ca93a86a846d5a9614d4e6c783011', path='experiment.yaml')
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]


# https://docs.azure.cn/en-us/databricks/_static/notebooks/mlflow/mlflow-end-to-end-example-azure.html
# wrappedModel = SklearnModelWrapper(model)
# # Log the model with a signature that defines the schema of the model's inputs and outputs. 
# # When the model is deployed, this signature will be used to validate inputs.
# signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
# mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, signature=signature)
