# data-science-template

This repo contians a template for ML/DS Projects.

The structure and documents were heavily influenced from:

- https://github.com/cmawer/reproducible-model
- https://github.com/drivendata/cookiecutter-data-science
- https://github.com/Azure/Azure-TDSP-ProjectTemplate

---

This project contains a very slimmed down version of python and R code that mimics a DS/ML project.

The code and Makefile will:

    - create python/R virtual environments
    - install package/library dependencies from the requirements.txt/DESCRIPTION files
    - run unit tests
    - run ETL scripts
    - run experiments using BayesSearchCV
    - run jupyter/R-markdown notesbooks that generate html/markdown outputs

The entire project can be built/ran with the simple command `make all` from the project directory. See `Makefile` for additional commands and implicit project DAG.

It requires Python 3.9 (which can be configured in the Makefile) and is currently ran with R 4.X.

---

To activate virtual environment run `source .venv/bin/activate`; for example:

```commandline
source .venv/bin/activate
jupyter notebook
```

## Repo structure 

```
├── README.md                  <- You are here
├── Makefile                   <- Makefile, which runs all components of the project with commands like `make all`, `make environment`, `make data`, etc.
├── requirements.txt           <- Python package dependencies
├── DESCRIPTION                <- R package dependencies
│
│
├── artifacts/                 <- All non-code/document artifacts (e.g. data, models, etc.).
│   ├── data/                  <- Folder that contains data (used or generated).
│       ├── external/          <- Data from third party sources.
│       ├── interim/           <- Intermediate data that has been transformed. (This directory is excluded via .gitignore)
│       ├── processed/         <- The final, canonical data sets for modeling. (This directory is excluded via .gitignore)
│       ├── raw/               <- The original, immutable data dump. (This directory is excluded via .gitignore)
│   ├── models/                <- Trained model objects (TMOs), model predictions, and/or model summaries
│       ├── archive/           <- Folder that contains old models.
│       ├── current/           <- The current model being used by the project.
│       ├── experiments/       <- Contains experimentss and experiments output (e.g. yaml/html showing performance of experiments.)
│
├── code/                      <- All source-code (e.g. SQL, python scripts, notebooks, unit-tests, etc.)
│   ├── config/                <- Directory for yaml configuration files for model training, scoring, etc
│   ├── notebooks/             <- All notebooks in projects. Users should attempt to separate reporting/exploring (notebooks) and processing (scripts).
│       ├── archive/           <- Previous notebooks that are outdated or no longer applicable.
│       ├── deliver/           <- Notebooks shared with others. 
│       ├── develop/           <- Current notebooks being used in development.
│       ├── template.ipynb     <- Template notebook for analysis with useful imports and helper functions. 
│   ├── scripts/               <- Source-code for the project. Should contain most of the processing, model development, and business logic. Code should be modular and reusable.
│       ├── helpers/           <- Helper scripts used in the main files files 
│       ├── sql/               <- 
│       ├── file.py            <- python script
│       ├── file.R             <- R script
│   ├── tests/                 <- Files necessary for running model tests (see documentation below) 
│       ├── test_files/        <- Files that help run unit tests, e.g. mock yaml files.
│       ├── test_file.py       <- python unit-test script
│       ├── test_file.R        <- R unit-test script
│
├── docs/                      <- All documentation, data dictionaries, manuals, and final reports and deliverables.
│   ├── data/                  <- Location to place documents describing results of data exploration, data dictionaries, etc.
│   ├── model/                 <- Model documentation 
│       ├── archive/
│       ├── baseline_model/
│       ├── current_model/
│   ├── project/               <- Project documentation, including project charter, and results.
│       ├── deliverables/      <- All generated and sharable deliverables.
│   ├── figures/               <- Centralized location for all figures and diagrams in project except for those embedded in notebooks.
│           ├── archive/
│           ├── data/
│           ├── models/
│
```

# Project Details

- see [docs/project/Charter.md](./docs/project/Charter.md) for project description.
- see [docs/project/Exit-Report.md](./docs/project/Exit-Report.md) for project results.
