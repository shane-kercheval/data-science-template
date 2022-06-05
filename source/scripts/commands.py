"""
This file contains the functions for the command line interface. The makefile calls the commands defined in
this file.

For help in terminal, navigate to the project directory, activate the virtual environment, and run the
following examples:
    - `python3.9 source/scripts/commands.py --help`
    - `python3.9 source/scripts/commands.py extract --help`
"""
import logging.config
import os
import sys

import click

sys.path.append(os.getcwd())
from source.library.utilities import Timer, log_func, log_info, get_config  # noqa
import source.scripts.experiments as experiments  # noqa
import source.scripts.etl as etl  # noqa


logging.config.fileConfig("source/config/logging_to_file.conf",
                          defaults={'logfilename': 'output/log.log'},
                          disable_existing_loggers=False)


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
    log_func("extract", params=dict(output_directory=output_directory))
    etl.extract(output_directory=output_directory)


@main.command()
def transform():
    """This function transforms the credit data."""
    config = get_config()
    input_directory = config['DATA']['RAW_DIRECTORY']
    output_directory = config['DATA']['PROCESSED_DIRECTORY']
    log_func("extract", params=dict(input_directory=input_directory, output_directory=output_directory))
    etl.transform(input_directory=input_directory, output_directory=output_directory)


@main.command()
@click.option('-n_iterations', default=50, help='the number of iterations for BayesSearchCV per model',
              show_default=True)
@click.option('-n_splits', default=10, help='the number of cross validation splits ', show_default=True)
@click.option('-n_repeats', default=1, help='the number of cross validation repeats', show_default=True)
@click.option('-score', default='roc_auc',
              help='A string identifying the score to evaluate model performance, e.g. `roc_auc`.',
              show_default=True)
def run_experiments(n_iterations, n_splits, n_repeats, score):
    """This function runs an ML experiment according to the parameters provided."""
    log_func("run-experiments")
    config = get_config()
    input_directory = config['DATA']['PROCESSED_DIRECTORY']
    output_directory = config['EXPERIMENTS']['DIRECTORY']

    results_directory = experiments.run(
        input_directory=input_directory,
        output_directory=output_directory,
        n_iterations=n_iterations,
        n_splits=n_splits,
        n_repeats=n_repeats,
        score=score
    )
    # save the timestamp of the experiments, which can be read in by the notebook template and then deleted
    file_path = os.path.join(config['NOTEBOOKS']['DIRECTORY'], 'new_results.txt')
    with open(file_path, 'w') as text_file:
        text_file.writelines(results_directory)


if __name__ == '__main__':
    main()
