
"""
This file contains the logic for running ML experiments using BayesSearchCV.
"""
import logging

import pandas as pd
from helpsk.sklearn_eval import MLExperimentResults
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

from source.service.model_registry import ModelRegistry
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
        n_iterations: int,
        n_splits: int,
        n_repeats: int,
        random_state: int,
        tags: str):

    registry = ModelRegistry(tracking_uri=tracking_uri)
    with registry.track_experiment(experiment_name=experiment_name, tags=tags) as tracker:
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
        tracker.log_model(model=bayes_search.best_estimator_)
        tracker.log_metric(metric=score, metric_value=bayes_search.best_score_)
        tracker.log_params(params=bayes_search.best_params_)
        tracker.log_ml_results(results=results, file_name='experiment.yaml')
        tracker.log_pickle(obj=x_train, file_name='x_train.pkl')
        tracker.log_pickle(obj=x_test, file_name='x_test.pkl')
        tracker.log_pickle(obj=y_train, file_name='y_train.pkl')
        tracker.log_pickle(obj=y_test, file_name='y_test.pkl')

    if registry.get_production_model(model_name=model_name) is None:
        tracker.last_run.put_model_in_production(model_name=model_name)
    else:
        production_run = registry.get_production_run(model_name=model_name)
        if bayes_search.best_score_ > production_run.metrics[score]:
            tracker.last_run.put_model_in_production(model_name=model_name)
