#################################################################################
# File adapted from https://github.com/drivendata/cookiecutter-data-science
#################################################################################
.PHONY: clean_python clean_r clean environment_python environment_r environment tests_python tests_r tests \
	data_extract data_transform data exploration_python exploration_r exploration experiment_1 \
	experiment_2 drift all

#################################################################################
# GLOBALS
#################################################################################
PYTHON_VERSION := 3.9
PYTHON_VERSION_SHORT := $(subst .,,$(PYTHON_VERSION))
PYTHON_INTERPRETER := python$(PYTHON_VERSION)
SNOWFLAKE_VERSION := 2.7.4

# PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))


FORMAT_MESSAGE =  "\n[MAKE "$(1)"] >>>" $(2)

#################################################################################
# Project-specific Commands
#################################################################################
tests_python: environment_python
	@echo $(call FORMAT_MESSAGE,"tests_python", "Running python unit tests.")
	rm -f source/tests/test_files/log.log
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover source/tests

linting:
	. .venv/bin/activate && flake8 --max-line-length 110 source/scripts
	. .venv/bin/activate && flake8 --max-line-length 110 source/library
	. .venv/bin/activate && flake8 --max-line-length 110 source/tests

tests_r: environment_r
	@echo $(call FORMAT_MESSAGE,"tests_r","Running R unit tests.")
	R --quiet -e "testthat::test_dir('source/tests')"

tests: tests_python tests_r
	@echo $(call FORMAT_MESSAGE,"tests","Finished running unit tests.")

## Make Dataset
data_extract:
	@echo $(call FORMAT_MESSAGE,"data_extract","Extracting data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/commands.py extract

data_transform:
	@echo $(call FORMAT_MESSAGE,"data_transform","Transforming data.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/commands.py transform

data: data_extract data_transform
	@echo $(call FORMAT_MESSAGE,"data","Finished running local ETL.")

exploration_python:
	@echo $(call FORMAT_MESSAGE,"exploration_python","Running exploratory jupyter notebooks and converting to .html files.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/data-profile.ipynb
	mv source/notebooks/data-profile.html output/data/data-profile.html

exploration_r: environment_r
	@echo $(call FORMAT_MESSAGE,"exploration_r","Running exploratory RMarkdown notebooks and converting to .md files.")
	Rscript -e "rmarkdown::render('source/notebooks/templates/r-markdown-template.Rmd')"
	rm -rf output/data/r-markdown-template_files/
	mv source/notebooks/templates/r-markdown-template.md output/data/r-markdown-template.md
	mv source/notebooks/templates/r-markdown-template_files/ output/data/

exploration: exploration_python exploration_r
	@echo $(call FORMAT_MESSAGE,"exploration","Finished running exploration notebooks.")

experiment_1:
	@echo $(call FORMAT_MESSAGE,"experiment_1","Running Hyper-parameters experiments based on BayesianSearchCV.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://localhost:1234' \
		-random_state=3

	@echo $(call FORMAT_MESSAGE,"experiment_1","Copying experiments template (experiment-template.ipynb) to /source/notebooks directory.")
	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_1.ipynb
	@echo $(call FORMAT_MESSAGE,"experiment_1","Running the notebook and creating html.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/experiment_1.ipynb
	mv source/notebooks/experiment_1.html output/experiment_1.html

experiment_2:
	@echo $(call FORMAT_MESSAGE,"experiment_2","Running Hyper-parameters experiments based on BayesianSearchCV.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://localhost:1234' \
		-random_state=42

	@echo $(call FORMAT_MESSAGE,"experiment_2","Copying experiments template (experiment-template.ipynb) to /source/notebooks directory.")
	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_2.ipynb
	@echo $(call FORMAT_MESSAGE,"experiment_2","Running the notebook and creating html.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/experiment_2.ipynb
	mv source/notebooks/experiment_2.html output/experiment_2.html

experiment_3:
	@echo $(call FORMAT_MESSAGE,"experiment_3","Running Hyper-parameters experiments based on BayesianSearchCV.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://localhost:1234' \
		-random_state=10

	@echo $(call FORMAT_MESSAGE,"experiment_3","Copying experiments template (experiment-template.ipynb) to /source/notebooks directory.")
	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_3.ipynb
	@echo $(call FORMAT_MESSAGE,"experiment_3","Running the notebook and creating html.")
	. .venv/bin/activate && jupyter nbconvert --execute --to html source/notebooks/experiment_3.ipynb
	mv source/notebooks/experiment_3.html output/experiment_3.html

experiments: experiment_1 experiment_2 experiment_3
	@echo $(call FORMAT_MESSAGE,"experiments","Done Running and Evaluating Experiments.")

drift: environment
	@echo $(call FORMAT_MESSAGE,"drift","Running evaluation of final model on test set.")

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: environment tests linting remove_logs data exploration experiments drift
	@echo $(call FORMAT_MESSAGE,"all","Finished running entire workflow.")

## Delete all generated files (e.g. virtual environment)
clean: clean_python clean_r mlflow_clean
	@echo $(call FORMAT_MESSAGE,"clean","Cleaning project files.")
	rm -f artifacts/data/raw/*.pkl
	rm -f artifacts/data/raw/*.csv
	rm -f artifacts/data/processed/*

mlflow_server:
	. .venv/bin/activate && \
		mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifact-root \
		--host 0.0.0.0 --port 1234

mlflow_ui:
	open http://127.0.0.1:1234

mlflow_kill:
	 pkill -f gunicorn

mlflow_clean:
	rm -rf mlruns
	rm -f mlflow.db
	rm -rf mlflow-artifact-root

#################################################################################
# Generic Commands
#################################################################################
clean_python:
	@echo $(call FORMAT_MESSAGE,"clean_python","Cleaning Python files.")
	rm -rf .venv
	find . \( -name __pycache__ \) -prune -exec rm -rf {} +
	find . \( -name .ipynb_checkpoints \) -prune -exec rm -rf {} +

clean_r:
	@echo $(call FORMAT_MESSAGE,"clean_r","Cleaning R files.")
	rm -rf renv
	rm -f .Rprofile

environment_python:
ifneq ($(wildcard .venv/.*),)
	@echo $(call FORMAT_MESSAGE,"environment_python","Found .venv directory. Skipping virtual environment creation.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo $(call FORMAT_MESSAGE,"environment_python","Did not find .venv directory. Creating virtual environment.")
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Creating environment at .venv.")
	@echo $(call FORMAT_MESSAGE,"environment_python","NOTE: Run this command to activate virtual environment: 'source .venv/bin/activate'.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Activating virtual environment.")
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing packages from requirements.txt.")
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt
	@echo $(call FORMAT_MESSAGE,"environment_python","Installing snowflake packages.")
	. .venv/bin/activate && pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v$(SNOWFLAKE_VERSION)/tested_requirements/requirements_$(PYTHON_VERSION_SHORT).reqs
	. .venv/bin/activate && pip install snowflake-connector-python==v$(SNOWFLAKE_VERSION)
	. .venv/bin/activate && brew install libomp
endif

environment_r:
ifneq ($(wildcard renv/.*),)
	@echo $(call FORMAT_MESSAGE,"environment_r","Found renv directory. Skipping virtual environment creation.")
else
	@echo $(call FORMAT_MESSAGE,"environment_r","Did not find renv directory. Creating virtual environment.")
	R --quiet -e 'install.packages("renv", repos = "http://cran.us.r-project.org")'
	# Creates `.Rprofile` file, and `renv` folder
	R --quiet -e 'renv::init(bare = TRUE)'
	R --quiet -e 'renv::install()'
endif

## Set up python/R virtual environments and install dependencies
environment: environment_python environment_r
	@echo $(call FORMAT_MESSAGE,"environment","Finished setting up environment.")

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
