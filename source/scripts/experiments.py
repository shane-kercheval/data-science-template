import datetime
import os
import sys
from pathlib import Path

from helpsk.sklearn_eval import MLExperimentResults
from helpsk.utility import read_pickle, to_pickle
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV  # noqa

sys.path.append(os.getcwd())
from source.library.utilities import Timer, log_info, log_func  # noqa
import source.library.classification_search_space as css  # noqa


def run(input_directory: str,
        output_directory: str,
        n_iterations: int,
        n_splits: int,
        n_repeats: int,
        score: str):
    """
    Logic For Running Experiments.
    """
    log_func("experiments.run", params=dict(
        input_directory=input_directory,
        output_directory=output_directory,
        n_iterations=n_iterations,
        n_splits=n_splits,
        n_repeats=n_repeats,
        score=score,
    ))

    log_info(f"Loading training/test sets from input directory {input_directory}")
    x_train = read_pickle(os.path.join(input_directory, 'X_train.pkl'))
    y_train = read_pickle(os.path.join(input_directory, 'y_train.pkl'))

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

    timestamp = f'{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}'
    file_name = f'multi-model-BayesSearchCV-{timestamp}'

    yaml_file_name = os.path.join(output_directory, f'{file_name}.yaml')
    model_file_name = os.path.join(output_directory, f'{file_name}_best_estimator.pkl')

    Path(output_directory).mkdir(exist_ok=True)

    log_info(f"Saving results of BayesSearchCV to: `{yaml_file_name}`")
    results.to_yaml_file(yaml_file_name=yaml_file_name)

    log_info(f"Saving the best_estimator of BayesSearchCV to: `{model_file_name}`")
    to_pickle(bayes_search.best_estimator_, model_file_name)

    log_info(f"Best Score: {bayes_search.best_score_}")
    log_info(f"Best Params: {bayes_search.best_params_}")

    with open(os.path.join(output_directory, 'new_results.txt'), 'w') as text_file:
        text_file.writelines(file_name)
