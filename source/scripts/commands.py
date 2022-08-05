"""
This file contains the functions for the command line interface. The makefile calls the commands
defined in this file.

For help in terminal, navigate to the project directory, activate the virtual environment, and run
the following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""
import logging.config
import click

import source.scripts.experiments as experiments
import source.scripts.etl as etl
from source.library.utility import get_config


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
    output_directory = get_config()['DATA']['RAW_DIRECTORY']
    etl.extract(output_directory=output_directory)


@main.command()
def transform():
    """This function transforms the credit data."""
    config = get_config()
    input_directory = config['DATA']['RAW_DIRECTORY']
    output_directory = config['DATA']['PROCESSED_DIRECTORY']
    etl.transform(input_directory=input_directory, output_directory=output_directory)


@main.command()
@click.option('-n_iterations', default=4,
              help='the number of iterations for BayesSearchCV per model',
              show_default=True)
@click.option('-n_splits', default=3,
              help='the number of cross validation splits ', show_default=True)
@click.option('-n_repeats', default=1,
              help='the number of cross validation repeats', show_default=True)
@click.option('-score', default='roc_auc',
              help='A string identifying the score to evaluate model performance, e.g. `roc_auc`.',
              show_default=True)
@click.option('-tracking_uri', default='http://localhost:1234',
              help='MLFlow tracking_uri',
              show_default=True)
@click.option('-required_performance_gain', default=0.03,
              help='The percent increase required to accept best model into production.',
              show_default=True)
@click.option('-random_state', default=42,
              help='Random state/seed to generate consistent results.',
              show_default=True)
def run_experiments(n_iterations: int,
                    n_splits: int,
                    n_repeats: int,
                    score: str,
                    tracking_uri: str,
                    required_performance_gain: float,
                    random_state):
    """This function runs an ML experiment according to the parameters provided."""
    config = get_config()
    experiments.run(
        input_directory=config['DATA']['PROCESSED_DIRECTORY'],
        n_iterations=n_iterations,
        n_splits=n_splits,
        n_repeats=n_repeats,
        score=score,
        tracking_uri=tracking_uri,
        experiment_name=config['MLFLOW']['EXPERIMENT_NAME'],
        registered_model_name=config['MLFLOW']['MODEL_NAME'],
        required_performance_gain=required_performance_gain,
        random_state=random_state,
    )


if __name__ == '__main__':
    main()
