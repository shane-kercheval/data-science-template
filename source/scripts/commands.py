"""
This file contains the functions for the command line interface.
"""
import logging.config
import os
import sys

import click

sys.path.append(os.getcwd())
from source.library.utilities import Timer, log_func, log_info  # noqa
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
    log_func("extract")
    etl.extract()


@main.command()
def transform():
    log_func("transform")
    etl.transform()


@main.command()
@click.option('-input_directory', default='', show_default=True)
@click.option('-output_directory', default='', show_default=True)
def create_training_test(input_directory, output_directory):
    log_func("create_training_test")
    assert input_directory != ''
    assert output_directory != ''
    etl.create_training_test(input_directory=input_directory, output_directory=output_directory)


@main.command()
@click.option('-input_directory', default='', show_default=True)
@click.option('-output_directory', default='', show_default=True)
@click.option('-n_iterations', default=50, show_default=True)
@click.option('-n_splits', default=10, show_default=True)
@click.option('-n_repeats', default=1, show_default=True)
@click.option('-score', default='roc_auc', show_default=True)
def run_experiments(input_directory, output_directory, n_iterations, n_splits, n_repeats, score):
    log_func("run-experiments")
    assert input_directory != ''
    assert output_directory != ''
    experiments.run(input_directory, output_directory, n_iterations, n_splits, n_repeats, score)


if __name__ == '__main__':
    main()
