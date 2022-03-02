# data-science-template

This repo is a template for ML/DS Projects. It's influenced from:

- https://github.com/cmawer/reproducible-model
- https://github.com/drivendata/cookiecutter-data-science


## Repo structure 

```
├── README.md                     <- You are here
│
├── config                        <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging                   <- Configuration of python loggers
│
├── data                          <- Folder that contains data used or generated.
│   ├── external                  <- Data from third party sources.
│   └── raw                       <- The original, immutable data dump. (This directory is excluded via .gitignore)
│   ├── interim                   <- Intermediate data that has been transformed. (This directory is excluded via .gitignore)
│   ├── processed                 <- The final, canonical data sets for modeling. (This directory is excluded via .gitignore)
│
├── deliverables                  <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── archive                   <- Previous deliverables that are outdated or no longer applicable.
│
├── figures                       <- Generated graphics and figures to be used in reporting and analysis.
│   ├── deliverables              <- Generated graphics and figures to be used in reporting.
│   ├── [etc]                     <- Various folders to organize figures (e.g. model evaluation).
│   ├── archive                   <- Previous figures that are outdated or no longer applicable.
│
├── models                        <- Trained model objects (TMOs), model predictions, and/or model summaries
│   ├── archive                   <- Previous models that are outdated or no longer applicable. (This directory is excluded via .gitignore)
│
├── notebooks
│   ├── develop                   <- Current notebooks being used in development.
│   ├── deliver                   <- Notebooks shared with others. 
│   ├── archive                   <- Previous notebooks that are outdated or no longer applicable.
│   ├── template.ipynb            <- Template notebook for analysis with useful imports and helper functions. 
│
├── references                    <- Data dictionaries, manuals, and all other explanatory materials.
│
├── src                           <- Source data for the sybil project 
│   ├── archive                   <- Previous scripts that are outdated or no longer applicable.
│   ├── helpers                   <- Helper scripts used in main src files 
│   ├── sql                       <- SQL source code
│   ├── ingest_data.py            <- Script for ingesting data from different sources 
│   ├── generate_features.py      <- Script for cleaning and transforming data and generating features used for use in training and scoring.
│   ├── train_model.py            <- Script for training machine learning model(s)
│   ├── score_model.py            <- Script for scoring new predictions using a trained model.
│   ├── postprocess.py            <- Script for postprocessing predictions and model results
│   ├── evaluate_model.py         <- Script for evaluating model performance 
│
├── tests                         <- Files necessary for running model tests (see documentation below) 
│   ├── test-files                <- Directory where artifacts and results of tests are saved to be compared to the sources of truth. Only .gitkeep in this directory should be synced to Github
│   ├── test.py                   <- Runs the tests defined in test_config.yml and then compares the produced artifacts/results with those defined as expected in the true/ directory
│
│
├── Makefile                      <- Makefile with commands like `make data` or `make train`
├── run.py                        <- Simplifies the execution of one or more of the src scripts 
├── requirements.txt              <- Python package dependencies 
```

The project structure was heavily influenced by - https://github.com/cmawer/reproducible-model.


# Project Requirements

> This section should describe the project-specific requirements (e.g. python3.9).

- python3.9
- R version XXX

# Project Overview

TBD
