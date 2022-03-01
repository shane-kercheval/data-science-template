#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: environment data data_extract data_transform clean

#################################################################################
# GLOBALS
#################################################################################
PYTHON_INTERPRETER = python3.9
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

#################################################################################
# Project-specific Commands
#################################################################################

## Make Dataset
data_extract: environment
	@echo ">>> Extracting data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/data.py extract

data_transform: environment
	@echo ">>> Transforming data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/data.py transform

data_training_test: environment
	@echo ">>> Transforming data."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/data.py create-training-test

data: data_extract data_transform data_training_test

exploration: environment
	. .venv/bin/activate && jupyter nbconvert --execute --to html notebooks/develop/Data-Exploration.ipynb

all: data exploration

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
	@echo ">>> Found .venv, skipping virtual environment creation."
	@echo ">>> Activating virtual environment."
	@echo ">>> Installing packages from requirements.txt."
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo ">>> Did not find .venv, creating virtual environment."
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv
	@echo ">>> Installing virtualenv."
	virtualenv .venv
	@echo ">>> NOTE: Creating environment at .venv."
	@echo ">>> NOTE: To activate virtual environment, run: 'source .venv/bin/activate'."
	@echo ">>> Activating virtual environment."
	@echo ">>> Installing packages from requirements.txt."
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
