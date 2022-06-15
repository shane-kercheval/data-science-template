FROM python:3.9

RUN apt-get update -y && apt-get install zsh -y
RUN PATH="$PATH:/usr/bin/zsh"

RUN mkdir /code
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/code"
