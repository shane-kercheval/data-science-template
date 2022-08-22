
"""
This file contains the logic for running ML experiments using BayesSearchCV.
"""
import datetime
import os
import logging

import pandas as pd
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
from source.service.model_registry import Tracker, ModelRegistry, MLStage, Run
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

    last_run = tracker.last_run
    # last_run = Run.load(
    #     experiment_name=experiment_name,
    #     run_name=tracker.last_run_name,
    #     registry=registry
    # )

    # if there is no model in production; put the current model in production
    # otherwise check if this score from this experiment is higher than the production score
    # if so, put this model into production
    production_model = registry.get_production_model(model_name=model_name)
    if production_model is None:
        model_version = last_run.register_model(model_name=model_name)
        registry.transition_model_to_stage(
            model_name=model_name,
            model_version=model_version.version,
            to_stage=MLStage.PRODUCTION,
        )
    else:
        production_run = Run.load_from_id(
            experiment_name=experiment_name,
            run_id=production_model.run_id,
            registery=registry
        )

        if bayes_search.best_score_ > production_run.metrics['score']:
            model_version = last_run.register_model(model_name=model_name)
            prod_version = registry.get_production_model(model_name=model_name)

            _ = registry.transition_model_to_stage(
                model_name=prod_version.name,
                model_version=prod_version.version,
                to_stage=MLStage.ARCHIVED
            )

            registry.transition_model_to_stage(
                model_name=model_name,
                model_version=model_version.version,
                to_stage=MLStage.PRODUCTION,
            )
