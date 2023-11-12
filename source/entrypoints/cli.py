"""
ontains the functions for the command line interface. The makefile calls the commands defined in
this file.

For help in terminal, navigate to the project directory, run the docker container, and from within
the container run the following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""

import logging.config
import logging
import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from helpsk.utility import read_pickle

from source.config import config
from source.domain import experiment
from source.service import etl, model_registry


logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False,
)


@click.group()
def main() -> None:
    """Logic For Extracting and Transforming Datasets."""
    pass


@main.command()
def extract() -> None:
    """Downloads the credit data from openml.org."""
    etl.extract(output_directory=config.dir_data_raw())


@main.command()
def transform() -> None:
    """Transforms the credit data."""
    etl.transform(
        input_directory=config.dir_data_raw(),
        output_directory=config.dir_data_processed(),
    )


@main.command()
@click.option('-n_iterations', default=4,
              help='the number of iterations for BayesSearchCV per model',
              show_default=True)
@click.option('-n_folds', default=3,
              help='the number of cross validation splits ', show_default=True)
@click.option('-n_repeats', default=1,
              help='the number of cross validation repeats', show_default=True)
@click.option('-score', default='roc_auc',
              help='A string identifying the score to evaluate model performance, e.g. `roc_auc`.',
              show_default=True)
@click.option('-required_performance_gain', default=0.03,
              help='The percent increase required to accept best model into production.',
              show_default=True)
@click.option('-random_state', default=42,
              help='Random state/seed to generate consistent results.',
              show_default=True)
def run_experiment(
        n_iterations: int,
        n_folds: int,
        n_repeats: int,
        score: str,
        required_performance_gain: float,
        random_state: int) -> None:
    """
    Runs an ML experiment according to the parameters provided.

    Args:
        n_iterations:
            The number of iterations that the BayesSearchCV will search per model.
            e.g. A value of 20 means that 20 different hyper-parameter combinations will be
            searched for each model.
        n_folds:
            The number of folds to use when cross-validating.
        n_repeats:
            The number of repeats to use when cross-validating.
        score:
            The name of the metric/score e.g. 'roc_auc'
        required_performance_gain:
            percent increase required to put newly trained model into production
            For example:
                - a value of 0 means that if the new model's performance is identical to (or better
                    than) the old model's performance, then put the new model into production
                - a value of 0.01 means that if the new model's performance is 1% higher (or more)
                    than the old model's performance, then put the new model into production
        random_state:
            Random state/seed to generate consistent results.
    """
    logging.info("Splitting training & test datasets")
    credit_data = read_pickle(os.path.join(config.dir_data_processed(), 'credit.pkl'))
    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')
    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    # keep random_state the same across experiments to compare apples to apples
    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full,
        test_size=0.2, random_state=42,
    )
    put_in_production, tracker = experiment.run_bayesian_search(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=config.experiment_server_url(),
        experiment_name=config.experiment_name(),
        model_name=config.model_name(),
        score=score,
        required_performance_gain=required_performance_gain,
        n_iterations=n_iterations,
        n_folds=n_folds,
        n_repeats=n_repeats,
        random_state=random_state,
        tags={'type': 'BayesSearchCV'},
    )
    logging.info(f"Finished running experiment ({round(tracker.elapsed_seconds)} seconds)")
    logging.info(f"Model was put into production: {put_in_production}")
    logging.info(f"Experiment id: {tracker.last_run.exp_id}")
    logging.info(f"Experiment name: {tracker.last_run.exp_name}")
    logging.info(f"Run id: {tracker.last_run.run_id}")
    logging.info(f"Metric(s): {tracker.last_run.metrics}")


@main.command()
@click.option('-input_file',
              default=os.path.join(config.dir_data_processed(), 'credit.pkl'),
              help='the path to the csv to make predictions on',
              show_default=True)
@click.option('-output_file',
              default=os.path.join(config.dir_ouput(), 'credit_predictions.csv'),
              help='the path to save the predictions (as csv)',
              show_default=True)
def predict(input_file: str, output_file: str) -> None:
    """
    Makes predictions on the input file and saves the predictions to the output file, based on the
    production model saved to the MLflow server from the training above.

    Args:
        input_file:
            The path to the csv to make predictions on.
        output_file:
            The path to save the predictions (as csv).
    """
    credit_data = pd.read_pickle(input_file)
    logging.info(f"Making predictions on {len(credit_data)} loans from {input_file}")
    registry = model_registry.ModelRegistry(tracking_uri=config.experiment_server_url())
    model_info = registry.get_production_model(model_name=config.model_name())
    model = registry.download_artifact(
        run_id=model_info.run_id,
        artifact_name='model/model.pkl',
        read_from=pd.read_pickle,
    )
    predictions = model.predict(credit_data)
    logging.info(f"Saving predictions to {output_file}")
    pd.DataFrame({'predictions': predictions}).to_csv(output_file)



if __name__ == '__main__':
    main()
