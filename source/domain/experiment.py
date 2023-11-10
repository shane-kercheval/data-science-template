
"""Contains the logic for running ML experiments using BayesSearchCV."""

import logging

import pandas as pd
from helpsk.sklearn_eval import MLExperimentResults
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

from source.service.model_registry import ModelRegistry, Tracker
import source.domain.classification_search_space as css


def run_bayesian_search(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        tracking_uri: str,
        experiment_name: str,
        model_name: str,
        score: str,
        required_performance_gain: float,
        n_iterations: int,
        n_folds: int,
        n_repeats: int,
        random_state: int,
        tags: dict[str, str]) -> tuple[bool, Tracker]:
    """
    Runs BayesSearchCV and searches the models and pre-processing steps defined in
    classification_search_space.create_search_space() against the training and test sets passed in.

    The experiment results and corresponding training/tests sets are saved to the mlflow server
    with the corresponding `tracking_uri` provided. The runs will be in an experiment called
    `experiment_name`. The best model found by BayesSearchCV will be registered as
    `registered_model_name` with a new version number. The best model will be put into production
    if it has a `required_performance_gain` percent increase compared with the current model in
    production.

    Args:
        x_train:
            training features
        x_test:
            test features
        y_train:
            training target
        y_test:
            test target
        tracking_uri:
            the tracking URI for the experimentation server
        experiment_name:
            The name of the experiment saved to MLFlow.
        model_name:
            The name of the model saved to MLFlow.
        score:
            The name of the metric/score e.g. 'roc_auc'
        required_performance_gain:
            percent increase required to put newly trained model into production
            For example:
                - a value of 0 means that if the new model's performance is identical to (or better
                than) the old model's performance, then put the new model into production
                - a value of 1 means that if the new model's performance is 1% higher (or more)
                    than the old model's performance, then put the new model into production
        n_iterations:
            The number of iterations that the BayesSearchCV will search per model.
            e.g. A value of 20 means that 20 different hyper-parameter combinations will be
            searched for each model.
        n_folds:
            The number of folds to use when cross-validating.
        n_repeats:
            The number of repeats to use when cross-validating.
        random_state:
            The random stage
        tags:
            A dictionary of tags to store in MLFlow.
    """
    registry = ModelRegistry(tracking_uri=tracking_uri)
    with registry.track_experiment(exp_name=experiment_name, tags=tags) as tracker:
        bayes_search = BayesSearchCV(
            estimator=css.create_pipeline(data=x_train),
            search_spaces=css.create_search_space(iterations=n_iterations),
            cv=RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state),
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
        tracker.log_model(model=bayes_search.best_estimator_)
        tracker.log_metric(metric=score, metric_value=bayes_search.best_score_)
        tracker.log_params(params=bayes_search.best_params_)
        tracker.log_ml_results(results=results, file_name='experiment.yaml')
        tracker.log_pickle(obj=x_train, file_name='x_train.pkl')
        tracker.log_pickle(obj=x_test, file_name='x_test.pkl')
        tracker.log_pickle(obj=y_train, file_name='y_train.pkl')
        tracker.log_pickle(obj=y_test, file_name='y_test.pkl')

    put_in_prod = False
    if registry.get_production_model(model_name=model_name) is None:
        tracker.last_run.put_model_in_production(model_name=model_name)
        put_in_prod = True
    else:
        production_run = registry.get_production_run(model_name=model_name)
        required_performance_gain = 1 + required_performance_gain
        if bayes_search.best_score_ >= production_run.metrics[score] * required_performance_gain:
            tracker.last_run.put_model_in_production(model_name=model_name)
            put_in_prod = True

    return put_in_prod, tracker
