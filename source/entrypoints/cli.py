"""
This file contains the functions for the command line interface. The makefile calls the commands
defined in this file.

For help in terminal, navigate to the project directory, run the docker container, and from within
the container run the following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""
import logging.config
import logging
import os
import click
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from helpsk.utility import read_pickle

import source.config.config as config
import source.domain.experiment as experiment
import source.service.etl as etl


logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    """This function downloads the credit data from openml.org."""
    etl.extract(output_directory=config.dir_data_raw())


@main.command()
def transform():
    """This function transforms the credit data."""
    etl.transform(
        input_directory=config.dir_data_raw(),
        output_directory=config.dir_data_processed()
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
        random_state):
    """This function runs an ML experiment according to the parameters provided."""
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
        test_size=0.2, random_state=42
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
        tags=dict(type='BayesSearchCV'),
    )
    logging.info(f"Finished running experiment ({round(tracker.elapsed_seconds)} seconds)")
    logging.info(f"Model was put into production: {put_in_production}")
    logging.info(f"Experiment id: {tracker.last_run.exp_id}")
    logging.info(f"Experiment name: {tracker.last_run.exp_name}")
    logging.info(f"Run id: {tracker.last_run.run_id}")
    logging.info(f"Metric(s): {tracker.last_run.metrics}")
    logging.info(f"Model name: {tracker.last_run.model_version.name}")
    logging.info(f"Model stage: {tracker.last_run.model_version.current_stage}")


if __name__ == '__main__':
    main()
