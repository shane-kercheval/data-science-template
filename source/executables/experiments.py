import datetime
import pickle

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import read_pickle, to_pickle

from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV  # noqa

from helpers.utilities import get_logger, Timer
import helpers.classification_search_space as css


def main():
    """
    Logic For Running Experiments.
    """
    logger = get_logger()
    logger.info(f"Starting experiments.")
    logger.info(f"Loading training/test sets.")

    x_train = read_pickle('artifacts/data/processed/X_train.pkl')
    y_train = read_pickle('artifacts/data/processed/y_train.pkl')

    with Timer("Running Model Experiments (BayesSearchCV)", logger):
        bayes_search = BayesSearchCV(
            estimator=css.create_pipeline(data=x_train),
            search_spaces=css.create_search_space(iterations=5),
            cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=42),
            scoring='roc_auc',
            refit=True,
            return_train_score=False,
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
        bayes_search.fit(x_train, y_train)
        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV',
            parameter_name_mappings=css.get_search_space_mappings(),
        )

    timestamp = f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
    file_name = f'multi-model-BayesSearchCV-{timestamp}'

    yaml_file_name = f'artifacts/models/experiments/{file_name}.yaml'
    model_file_name = f'artifacts/models/experiments/{file_name}_best_estimator.pkl'

    logger.info(f"Saving results of BayesSearchCV to: `{yaml_file_name}`")
    results.to_yaml_file(yaml_file_name=yaml_file_name)

    logger.info(f"Saving the best_estimator of BayesSearchCV to: `{model_file_name}`")
    to_pickle(bayes_search.best_estimator_, model_file_name)

    logger.info(f"Best Score: {bayes_search.best_score_}")
    logger.info(f"Best Params: {bayes_search.best_params_}")

    with open('artifacts/models/experiments/new_results.txt', 'w') as text_file:
        text_file.writelines(file_name)


if __name__ == '__main__':
    main()
