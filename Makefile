####
# DOCKER
####
docker_compose:
	# docker build -t data-science-template .
	docker compose -f docker-compose.yml up --build

docker_run: notebook mlflow_ui zsh

notebook:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'

zsh:
	docker exec -it data-science-template-bash-1 /bin/zsh

####
# MLFLOW
####
mlflow_ui:
	open 'http://127.0.0.1:1235'

mlflow_kill:
	 pkill -f gunicorn

mlflow_clean:
	rm -rf mlruns
	rm -f mlflow.db
	rm -rf mlflow-artifact-root
	rm -rf mlflow_server/1235

####
# Project
####
tests:
	rm -f source/tests/test_files/log.log
	python -m unittest discover source/tests

linting:
	flake8 --max-line-length 110 source/scripts
	flake8 --max-line-length 110 source/library
	flake8 --max-line-length 110 source/tests

data_extract:
	python source/scripts/commands.py extract

data_transform:
	python source/scripts/commands.py transform

data: data_extract data_transform

exploration:
	jupyter nbconvert --execute --to html source/notebooks/data-profile.ipynb
	mv source/notebooks/data-profile.html output/data/data-profile.html

experiment_1:
	python source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://mlflow_server:1235' \
		-random_state=3
	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_1.ipynb
	jupyter nbconvert --execute --to html source/notebooks/experiment_1.ipynb
	mv source/notebooks/experiment_1.html output/experiment_1.html

experiment_2:
	python source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://mlflow_server:1235' \
		-random_state=42

	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_2.ipynb
	jupyter nbconvert --execute --to html source/notebooks/experiment_2.ipynb
	mv source/notebooks/experiment_2.html output/experiment_2.html

experiment_3:
	python source/scripts/commands.py run-experiments \
		-n_iterations=4 \
		-n_splits=3 \
		-n_repeats=1 \
		-score='roc_auc' \
		-tracking_uri='http://mlflow_server:1235' \
		-random_state=10

	cp source/notebooks/templates/experiment-template.ipynb source/notebooks/experiment_3.ipynb
	jupyter nbconvert --execute --to html source/notebooks/experiment_3.ipynb
	mv source/notebooks/experiment_3.html output/experiment_3.html

experiments: experiment_1 experiment_2 experiment_3

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: tests linting remove_logs data exploration experiments

## Delete all generated files (e.g. virtual)
clean: mlflow_clean
	rm -f data/raw/*.pkl
	rm -f data/raw/*.csv
	rm -f data/processed/*
