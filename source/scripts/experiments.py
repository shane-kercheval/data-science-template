"""
This file contains the logic for running ML experiments using BayesSearchCV.
"""
import datetime
import os
import logging

import mlflow
import mlflow.exceptions
from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import read_pickle
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skopt import BayesSearchCV

from helpsk.logging import log_function_call, log_timer
import source.library.ml as ml
import source.library.classification_search_space as css


@log_function_call
@log_timer
def run(input_directory: str,
        n_iterations: int,
        n_splits: int,
        n_repeats: int,
        score: str,
        tracking_uri: str,
        experiment_name: str,
        registered_model_name: str,
        required_performance_gain: float = 0.02,
        random_state: int = 42):
    """
    Logic For Running Experiments. This function takes the full credit dataset in `input_directory`, and
    separates the dataset into training and test sets. The training set is used by BayesSearchCV to search for
    the best model.

    The experiment results and corresponding training/tests sets are saved to the mlflow server with the
    corresponding `tracking_uri` provided. The runs will be in an experiment called `experiment_name`.
    The best model found by BayesSearchCV will be registered as `registered_model_name` with a new version
    number. The best model will be put into production if it has a `required_performance_gain` percent
    increase compared with the current model in production.

    Args:
        input_directory:
            the directory to find credit.pkl
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
            The required performance gain, as percentage, compared with current model in production, required
            to put the new model (i.e. best model found by BayesSearchCV) into production. The default value
            is `0.02` meaning if the best model found by BayesSearchCV is >=2% better than the current
            model's performance, we will archive the current model in production and put the new model
            in production.
        random_state:
            random_state to pass to `BayesSearchCV`
    """
    timestamp = f'{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}'
    logging.info("Splitting training & test datasets")
    credit_data = read_pickle(os.path.join(input_directory, 'credit.pkl'))
    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')
    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    # keep random_state the same across experiments to compare apples to apples
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    ml.initialize_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    with mlflow.start_run(run_name=timestamp,
                          description=timestamp,
                          tags=dict(type='BayesSearchCV')):

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
        logging.info(f"Best Score: {bayes_search.best_score_}")
        logging.info(f"Best Params: {bayes_search.best_params_}")

        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV',
            parameter_name_mappings=css.get_search_space_mappings(),
        )
        assert bayes_search.scoring == score

        wrapped_model = ml.SklearnModelWrapper(bayes_search.best_estimator_)
        # Log the model with a signature that defines the schema of the model's inputs and outputs.
        # When the model is deployed, this signature will be used to validate inputs.
        mlflow.sklearn.log_model(
            sk_model=wrapped_model,
            artifact_path='model',
        )
        mlflow.log_metric(score, bayes_search.best_score_)
        params = bayes_search.best_params_.copy()
        _ = params.pop('model', None)
        mlflow.log_params(params=params)
        ml.log_ml_results(results=results, file_name='experiment.yaml')
        ml.log_pickle(obj=x_train, file_name='x_train.pkl')
        ml.log_pickle(obj=x_test, file_name='x_test.pkl')
        ml.log_pickle(obj=y_train, file_name='y_train.pkl')
        ml.log_pickle(obj=y_test, file_name='y_test.pkl')

    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        production_model = client.get_latest_versions(name=registered_model_name, stages=['Production'])
    except mlflow.exceptions.RestException:
        production_model = []

    if len(production_model) == 0:
        # we don't currently have a model in production so put current model into production
        logging.info("No models currently in production.")
        model_version = ml.transition_last_model(
            ml_client=client,
            model_name=registered_model_name,
            stage='Production',
        )
        logging.info(
            f"Transitioning `{registered_model_name}` "
            "(version {model_version.version}) to Production"
        )
    else:
        # if the new model outperforms the current model in production, then
        # put the new model into production after archiving model currently in production
        assert len(production_model) == 1  # can only have one model in production
        production_model = production_model[0]
        production_score = client.get_run(run_id=production_model.run_id).data.metrics[score]
        if bayes_search.best_score_ > production_score * (1 + required_performance_gain):
            logging.info(
                f"Archiving previous model "
                f"(version {production_model.version}; {score} {production_score}) and placing new "
                f"model into Production ({score} {bayes_search.best_score_})"
            )
            _ = client.transition_model_version_stage(
                name=production_model.name,
                version=production_model.version,
                stage='Archived'
            )
            model_version = ml.transition_last_model(
                ml_client=client,
                model_name=registered_model_name,
                stage='Production',
            )
            logging.info(
                f"Transitioning `{registered_model_name}` "
                f"(version {model_version.version}) to Production"
            )
        else:
            logging.info(
                f"New Score: {score} - {bayes_search.best_score_} vs "
                f"Current Production Score: {score} - {production_score}); Keeping Production Model"
            )

    return timestamp
