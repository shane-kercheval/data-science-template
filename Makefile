#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: clean_python clean_r clean environment_python environment_r environment tests_python tests_r tests data_extract data_transform data_training_test data exploration_python exploration_r exploration experiments experiments_eval final_model final_eval all

#################################################################################
# GLOBALS
#################################################################################
PYTHON_VERSION := 3.9
PYTHON_VERSION_SHORT := $(subst .,,$(PYTHON_VERSION))
PYTHON_INTERPRETER := python$(PYTHON_VERSION)
SNOWFLAKE_VERSION := 2.7.4

# PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# Project-specific Commands
#################################################################################
tests_python: environment_python
	@echo "[MAKE tests_python]>>> Running python unit tests."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover tests

tests_r: environment_r
	@echo "[MAKE tests_r]>>> Running R unit tests."
	R --quiet -e "testthat::test_dir('tests')"

tests: tests_python tests_r
	@echo "[MAKE tests]>>> Finished running unit tests."

## Make Dataset
data_extract: environment_python
	@echo "[MAKE data_extract]>>> Extracting data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py extract

data_transform: environment_python
	@echo "[MAKE data_transform]>>> Transforming data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py transform

data_training_test: environment_python
	@echo "[MAKE data_training_test]>>> Creating training & test sets."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py create-training-test

data: data_extract data_transform data_training_test
	@echo "[MAKE data]>>> Finished running local ETL."

exploration_python: environment_python data_training_test
	@echo "[MAKE exploration_python]>>> Running exploratory jupyter notebooks and converting to .html files."
	. .venv/bin/activate && jupyter nbconvert --execute --to html notebooks/develop/Data-Exploration.ipynb

exploration_r: environment_r
	@echo "[MAKE exploration_r]>>> Running exploratory RMarkdown notebooks and converting to .html files."
	Rscript -e "rmarkdown::render('notebooks/develop/r-markdown-template.Rmd')"

exploration: exploration_python exploration_r
	@echo "[MAKE exploration]>>> Finished running exploration notebooks."

experiments: environment
	@echo "[MAKE experiments]>>> Running Hyper-parameters experiments based on BayesianSearchCV."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/run_experiments.py

experiments_eval: models/experiments/new_results.txt
	@echo "[MAKE experiments_eval]>>> Running Evaluation of experiments"
	@echo "[MAKE experiments_eval]>>> Moving experimentation .ipynb template to models/experiments directory."
	cp notebooks/experiment-template.ipynb notebooks/develop/$(shell cat models/experiments/new_results.txt).ipynb
	@echo "[MAKE experiments_eval]>>> Setting the experiments yaml file name within the ipynb file."
	sed -i '' 's/XXXXXXXXXXXXXXXX/$(shell cat models/experiments/new_results.txt)/g' notebooks/develop/$(shell cat models/experiments/new_results.txt).ipynb
	@echo "[MAKE experiments_eval]>>> Running the notebook and creating html."
	. .venv/bin/activate && jupyter nbconvert --execute --to html notebooks/develop/$(shell cat models/experiments/new_results.txt).ipynb
	rm -f models/experiments/new_results.txt

final_model: environment
	@echo "[MAKE final_model]>>> Building final model from best model in experiment."

final_eval: environment
	@echo "[MAKE final_eval]>>> Running evaluation of final model on test set."

## Run entire workflow.
all: environment tests data exploration experiments experiments_eval final_model final_eval
	@echo "[MAKE all]>>> Finished running entire workflow."

## Delete all generated files (e.g. virtual environment)
clean: clean_python clean_r
	@echo "[MAKE clean]>>> Cleaning project files."
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*

#################################################################################
# Generic Commands
#################################################################################
clean_python:
	@echo "[MAKE clean_python]>>> Cleaning Python files."
	rm -rf .venv
	find . -type d -name "__pycache__" -delete

clean_r:
	@echo "[MAKE clean_r]>>> Cleaning R files."
	rm -rf renv
	rm -f .Rprofile

environment_python:
ifneq ($(wildcard .venv/.*),)
	@echo "[MAKE environment_python]>>> Found .venv, skipping virtual environment creation."
	@echo "[MAKE environment_python]>>> Activating virtual environment."
	@echo "[MAKE environment_python]>>> Installing packages from requirements.txt."
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo "[MAKE environment_python]>>> Did not find .venv, creating virtual environment."
	python -m pip install --upgrade pip
	python -m pip install -q virtualenv
	@echo "[MAKE environment_python]>>> Installing virtualenv."
	virtualenv .venv --python=$(PYTHON_INTERPRETER)
	@echo "[MAKE environment_python]>>> NOTE: Creating environment at .venv."
	@echo "[MAKE environment_python]>>> NOTE: To activate virtual environment, run: 'source .venv/bin/activate'."
	@echo "[MAKE environment_python]>>> Activating virtual environment."
	@echo "[MAKE environment_python]>>> Installing packages from requirements.txt."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "[MAKE environment_python]>>> Installing snowflake packages."
	. .venv/bin/activate && pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v$(SNOWFLAKE_VERSION)/tested_requirements/requirements_$(PYTHON_VERSION_SHORT).reqs
	. .venv/bin/activate && pip install snowflake-connector-python==v$(SNOWFLAKE_VERSION)
endif

environment_r:
ifneq ($(wildcard renv/.*),)
	@echo "[MAKE environment_r]>>> Found renv, skipping virtual environment creation."
else
	@echo "[MAKE environment_r]>>> Did not find renv, creating virtual environment."
	R --quite -e 'install.packages("renv", repos = "http://cran.us.r-project.org")'
	# Creates `.Rprofile` file, and `renv` folder
	R --quite -e 'renv::init(bare = TRUE)'
	R --quite -e 'renv::install()'
endif

## Set up python/R virtual environments and install dependencies
environment: environment_python environment_r
	@echo "[MAKE environment]>>> Finished setting up environment."

#################################################################################
# Self Documenting Commands
#################################################################################
.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
