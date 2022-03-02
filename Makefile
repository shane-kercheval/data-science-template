#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: environment tests data data_extract data_transform clean exploration experiments experiments_eval final_model final_eval python_exploration r_exploration

#################################################################################
# GLOBALS
#################################################################################
PYTHON_INTERPRETER = python3.9
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# Project-specific Commands
#################################################################################

tests: environment
	@echo "[MAKE tests]>>> Running unit tests."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover tests

## Make Dataset
data_extract: environment
	@echo "[MAKE data_extract]>>> Extracting data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py extract

data_transform: environment
	@echo "[MAKE data_transform]>>> Transforming data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py transform

data_training_test: environment
	@echo "[MAKE data_training_test]>>> Creating training & test sets."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/etl.py create-training-test

data: data_extract data_transform data_training_test
	@echo "[MAKE data]>>> Running local ETL."


python_exploration: data_training_test
	@echo "[MAKE python_exploration]>>> Running exploratory jupyter notebooks and converting to .html files."
	. .venv/bin/activate && jupyter nbconvert --execute --to html notebooks/develop/Data-Exploration.ipynb

r_exploration: environment
	@echo "[MAKE r_exploration]>>> Running exploratory RMarkdown notebooks and converting to .html files."

exploration: python_exploration r_exploration
	@echo "[MAKE exploration]>>> Finished running exploration notebooks."

experiments: environment
	@echo "[MAKE experiments]>>> Running Hyper-parameters experiments based on BayesianSearchCV."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/run_experiments.py

experiments_eval: environment
	@echo "[MAKE experiments_eval]>>> Running Evaluation of experiments"

final_model: environment
	@echo "[MAKE final_model]>>> Building final model from best model in experiment."

final_eval: environment
	@echo "[MAKE final_eval]>>> Running evaluation of final model on test set."

all: tests data exploration experiments experiments_eval final_model final_eval

## Delete all generated files (e.g. virtual environment)
clean:
	rm -rf .venv
	rm -f data/raw/credit.pkl
	rm -f data/processed/X_test.pkl
	rm -f data/processed/X_train.pkl
	rm -f data/processed/y_test.pkl
	rm -f data/processed/y_train.pkl
	find . -type d -name "__pycache__" -delete

#################################################################################
# Generic Commands
#################################################################################

## Set up python virtual environment and install python dependencies
environment:
ifneq ($(wildcard .venv/.*),)
	@echo "[MAKE environment]>>> Found .venv, skipping virtual environment creation."
	@echo "[MAKE environment]>>> Activating virtual environment."
	@echo "[MAKE environment]>>> Installing packages from requirements.txt."
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo "[MAKE environment]>>> Did not find .venv, creating virtual environment."
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv
	@echo "[MAKE environment]>>> Installing virtualenv."
	virtualenv .venv
	@echo "[MAKE environment]>>> NOTE: Creating environment at .venv."
	@echo "[MAKE environment]>>> NOTE: To activate virtual environment, run: 'source .venv/bin/activate'."
	@echo "[MAKE environment]>>> Activating virtual environment."
	@echo "[MAKE environment]>>> Installing packages from requirements.txt."
	. .venv/bin/activate && pip install -r requirements.txt
endif

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
