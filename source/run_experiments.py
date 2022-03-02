import datetime
import pickle

import helpsk as hlp
from helpsk.sklearn_search_bayesian import BayesianSearchSpace
from helpsk.sklearn_search_bayesian_classification import LogisticBayesianSearchSpace, RandomForestBayesianSearchSpace
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

from helpers.utilities import get_logger, Timer


def main():
    """
    Logic For Running Experiments.
    """
    with open('data/processed/X_train.pkl', 'rb') as handle:
        x_train = pickle.load(handle)

    with open('data/processed/y_train.pkl', 'rb') as handle:
        y_train = pickle.load(handle)

    search_space = BayesianSearchSpace(
        x_train,
        model_search_spaces=[
            LogisticBayesianSearchSpace(iterations=3),
            RandomForestBayesianSearchSpace(iterations=3),
        ]
    )
    logger = get_logger()
    with Timer("Model Experiment (BayesSearchCV)", logger):
        logger.info("Starting BayesSearchCV")
        bayes_search = BayesSearchCV(
            estimator=search_space.pipeline(),
            search_spaces=search_space.search_spaces(),
            cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=42),
            scoring='roc_auc',
            refit=False,
            return_train_score=False,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        bayes_search.fit(x_train, y_train)
        results = hlp.sklearn_eval.MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV',
            parameter_name_mappings=search_space.param_name_mappings()
        )
        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        results_file = f'models/experiments/Multi-model-BayesSearchCV-{timestamp}.yaml'
        logger.info(f"Saving results of BayesSearchCV to: `{results_file}`")
        results.to_yaml_file(yaml_file_name=results_file)

        logger.info(f"Best Score: {bayes_search.best_score_}")
        logger.info(f"Best Params: {bayes_search.best_params_}")


if __name__ == '__main__':
    main()
