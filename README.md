# data-science-template

This repo contains a template for ML/DS Projects.

The structure and documents were heavily influenced from:

- https://github.com/cmawer/reproducible-model
- https://github.com/drivendata/cookiecutter-data-science
- https://github.com/Azure/Azure-TDSP-ProjectTemplate

---

This project contains python that mimics a very slimmed down version of a DS/ML project, which includes

- docker
- unit tests
- linting
- command line program for:
    - ETL
    - running ML experiments
- MLFlow
    - tracking experiments
    - registering models
    - transitioning models to production

The `docker-compose.yml` files define a basic image built on the `shanekercheval/python:ml` image. The compose file contains three services, MLFlow, Jupyter, and shell.

NOTE: The `requirements.txt` is not used by docker, but it is used by Github Workflows for running unit-tests.

# Running the Project

## Starting Docker

Build and run docker-compose:

```
make docker_compose
```

Open the MLFlow Client/UI and Jupyter Notebook in the browser, and start a terminal session (via zsh) in the container. This command will open MLFlow and Jupyter to the default browser, and will connect to the container's terminal in the same terminal window.

```
make docker_run
```

## Running the Code

The `Makefile` runs all components of the project. You can think of it as containing the implicit DAG, or recipe, of the project.

**Run make commands from terminal connected to container via `make docker_run` or `make zsh`**.

If you want to run the entire project from start to finish, including unit tests and linting, run:

```
make all
```

Common commands available from the Makefile are:

- `make all`: The entire project can be built/ran with the simple command `make all` from the project directory, which runs all components (build virtual environments, run tests, run scripts, generate output, etc.)
- `make clean`: Removes all virtual environments, python/R generated/hidden folders, and interim/processed data.
- `make environment`: Creates python/R virtual environments and install packages from the requirements.txt/DESCRIPTION files
- `make data`: Runs ETL scripts
- `make exploration`: Runs exploration notebooks and generate html/md documents.
- `make experiments`: Runs scripts which use BayesSearchCV over several models.
- `make experiments_eval`: Runs notebooks which evaluates the performance of the BayesSearchCV and produces an html report.
- `make final_model`: Retrains the best model from the most recent experiments on all data, and predict on test/holdout set. (not implemented yet)
- `make final_eval`: Runs the notebook which shows the performance of the final model and produces an html report. (not implemented yet)

See `Makefile` for additional commands and implicit project DAG.

---

## Repo structure 

```
????????? README.md                  <- You are here
????????? Makefile                   <- Makefile, which runs all components of the project with commands like `make all`, `make environment`, `make data`, etc.
????????? requirements.txt           <- Python package dependencies
???
???
????????? data/                  <- Folder that contains data (used or generated).
???   ????????? external/          <- Data from third party sources.
???   ????????? interim/           <- Intermediate data that has been transformed. (This directory is excluded via .gitignore)
???   ????????? processed/         <- The final, canonical data sets for modeling. (This directory is excluded via .gitignore)
???   ????????? raw/               <- The original, immutable data dump. (This directory is excluded via .gitignore)
???
????????? source/                    <- All source-code (e.g. SQL, python scripts, notebooks, unit-tests, etc.)
???   ????????? config/                <- Directory for yaml configuration files for model training, scoring, etc
???   ????????? library/               <- Supporting source-code that promotes code reusability and unit-testing. Clients that use this code are notebooks, executables, and tests.
???   ????????? scripts/               <- command-line programs that execute the project tasks (e.g. etl & data processing, experiments, model-building, etc.). They typically have outputs that are artifacts (e.g. .pkl models or data).
???   ????????? notebooks/             <- Notebooks (e.g. Jupyter/R-Markdown)
???   ????????? sql/                   <- SQL scripts for querying DWH/lake. 
???   ????????? tests/                 <- Files necessary for running model tests (see documentation below) 
???       ????????? test_files/        <- Files that help run unit tests, e.g. mock yaml files.
???       ????????? test_file.py       <- python unit-test script
???
????????? output/                      <- All documentation, data dictionaries, manuals, and final reports and deliverables.
???   ????????? data/                  <- Location to place documents describing results of data exploration, data dictionaries, etc.
???   ????????? deliverables/          <- All generated and sharable deliverables.
???   ????????? models/                <- Model documentation 
???   ????????? figures/               <- Centralized location for all figures and diagrams in project except for those embedded in notebooks.
???           ????????? archive/
???           ????????? data/
???           ????????? models/
```

---

# Project Details

- see [docs/project/Charter.md](./docs/project/Charter.md) for project description.
- see [docs/project/Exit-Report.md](./docs/project/Exit-Report.md) for project results.
