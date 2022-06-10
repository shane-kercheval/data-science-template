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
        tracking_uri: str,
        experiment_name: str = 'credit',
        registered_model_name: str = 'credit_model',
        required_performance_gain: float = 0.025,
        random_state: int = 42) -> str:
    """
    Logic For Running Experiments. The timestamp of the experiment is returned by the function.

    This function takes the full credit dataset in `input_directory`, and separates the dataset into training
    and test sets. The training set is used by BayesSearchCV to search for the best model.
    
    The experiment results and corresponding training/tests sets  are saved to the mlflow server with the
    corresponding `tracking_uri` provided. The runs will be in an experiment called `experiment_name`. 
    The best model found by BayesSearchCV will be registered as `registered_model_name` with a new version
    number. The best model will be put into production if it has a `required_performance_gain` percent
    increase compared with the current model in production.


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
        experiment_name:
            The name of the experiment saved to MLFlow.
        registered_model_name:
            The name of the model to register with MLFlow.
        required_performance_gain:
            The required perforance gain, as percentage, compared with current model in production, required
            to put the new model (i.e. best model found by BayesSearchCV) into production. The default value
            is `0.025` meaning if the best model found by BayesSearchCV is >=2.5% better than the current
            model's performance, we will archive the current model in production and put the new model
            in production.
        random_state:
            random_state to pass to `train_test_split` and `BayesSearchCV`
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
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2,
                                                        random_state=random_state)

    # set up MLFlow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(registered_model_name=registered_model_name)
    with Timer("Running Model Experiments (BayesSearchCV)"):
        with mlflow.start_run(run_name=timestamp,
                              description=timestamp,
                              tags=dict(type='BayesSearchCV',
                                        timestamp=timestamp)):

            bayes_search = BayesSearchCV(
                estimator=css.create_pipeline(data=x_train),
                search_spaces=css.create_search_space(iterations=n_iterations),
                cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state),
                scoring=score,
                refit=True,
                return_train_score=False,
                n_jobs=-1,
                verbose=1,
                random_state=random_state,
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
            yaml_file_name = f'{file_name}.yaml'
            model_file_name = f'{file_name}_best_estimator.pkl'

            ml.log_ml_results(results=results, file_name=yaml_file_name)
            ml.log_pickle(obj=bayes_search.best_estimator_, file_name=model_file_name)
            ml.log_pickle(obj=x_train, file_name='x_train.pkl')
            ml.log_pickle(obj=x_test, file_name='x_test.pkl')
            ml.log_pickle(obj=y_train, file_name='y_train.pkl')
            ml.log_pickle(obj=y_test, file_name='y_test.pkl')

    # if we find a model that outperforms the current model in production by `1.025`

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
        production_score = client.get_run(run_id=production_model.run_id).data.metrics[score]
        if bayes_search.best_score_ > production_score * (1 + required_performance_gain):
            log_info(f"Archiving previous model "
                     f"(version {production_model.version}; {score} {production_score}) and placing new "
                     f"model into Production ({score} {bayes_search.best_score_})")
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
