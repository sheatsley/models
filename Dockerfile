# Dockerfile for Deep Learning Models: https://github.com/sheatsley/models
FROM sheatsley/datasets
COPY . /models
RUN cd /models && pip install -e .
