"""
This file contains the logic for running ML experiments using BayesSearchCV.
"""
import datetime
import os
import sys
from pathlib import Path

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import read_pickle, to_pickle
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV  # noqa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

sys.path.append(os.getcwd())
from source.library.utilities import Timer, log_info, log_func  # noqa
import source.library.classification_search_space as css  # noqa


def run(input_directory: str,
        output_directory: str,
        n_iterations: int,
        n_splits: int,
        n_repeats: int,
        score: str) -> str:
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
            A string identifying the score to evaluate model performance, e.g. `roc_auc`.
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
    output_directory = os.path.join(output_directory, f"experiment__{timestamp}")
    log_info(f"Saving data and results to `{output_directory}`")
    Path(output_directory).mkdir(exist_ok=True)

    log_info("Splitting training & test datasets")

    credit_data = read_pickle(os.path.join(input_directory, 'credit.pkl'))

    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')

    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    file_name = os.path.join(output_directory, 'x_train.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(x_train, file_name)

    file_name = os.path.join(output_directory, 'x_test.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(x_test, file_name)

    file_name = os.path.join(output_directory, 'y_train.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(y_train, file_name)

    file_name = os.path.join(output_directory, 'y_test.pkl')
    log_info(f"Creating `{file_name}`")
    to_pickle(y_test, file_name)

    with Timer("Running Model Experiments (BayesSearchCV)"):
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
        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV',
            parameter_name_mappings=css.get_search_space_mappings(),
        )

    file_name = 'experiment'
    yaml_file_name = os.path.join(output_directory, f'{file_name}.yaml')
    model_file_name = os.path.join(output_directory, f'{file_name}_best_estimator.pkl')

    log_info(f"Saving results of BayesSearchCV to: `{yaml_file_name}`")
    results.to_yaml_file(yaml_file_name=yaml_file_name)

    log_info(f"Saving the best_estimator of BayesSearchCV to: `{model_file_name}`")
    to_pickle(bayes_search.best_estimator_, model_file_name)

    log_info(f"Best Score: {bayes_search.best_score_}")
    log_info(f"Best Params: {bayes_search.best_params_}")

    return timestamp
