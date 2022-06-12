image:
	docker build -t data-science-template .

compose:
	docker compose -f docker-compose.yml up --build


notebook:
	open 'http://127.0.0.1:8888/?token=d4484563805c48c9b55f75eb8b28b3797c6757ad4871776d'


mlflow_ui:
	open 'http://127.0.0.1:1235'



mlflow_server:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifact-root \
		--host 0.0.0.0 --port 1234

mlflow_kill:
	 pkill -f gunicorn

mlflow_clean:
	rm -rf mlruns
	rm -f mlflow.db
	rm -rf mlflow-artifact-root
	rm -rf mlflow_server/1235



tests_python: environment_python
	rm -f source/tests/test_files/log.log
	python -m unittest discover source/tests

linting:
	flake8 --max-line-length 110 source/scripts
	flake8 --max-line-length 110 source/library
	flake8 --max-line-length 110 source/tests

tests_r: environment_r
	R --quiet -e "testthat::test_dir('source/tests')"

tests: tests_python tests_r

## Make Dataset
data_extract:
	python source/scripts/commands.py extract

data_transform:
	python source/scripts/commands.py transform

data: data_extract data_transform

exploration_python:
	jupyter nbconvert --execute --to html source/notebooks/data-profile.ipynb
	mv source/notebooks/data-profile.html output/data/data-profile.html

exploration_r: environment_r
	Rscript -e "rmarkdown::render('source/notebooks/templates/r-markdown-template.Rmd')"
	rm -rf output/data/r-markdown-template_files/
	mv source/notebooks/templates/r-markdown-template.md output/data/r-markdown-template.md
	mv source/notebooks/templates/r-markdown-template_files/ output/data/

exploration: exploration_python exploration_r

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

final_model: environment

final_eval: environment

remove_logs:
	rm -f output/log.log

## Run entire workflow.
all: environment tests linting remove_logs data exploration experiments final_model final_eval

## Delete all generated files (e.g. virtual environment)
clean: clean_python clean_r mlflow_clean
	rm -f artifacts/data/raw/*.pkl
	rm -f artifacts/data/raw/*.csv
	rm -f artifacts/data/processed/*
