# Dockerfile for Deep Learning Models: https://github.com/sheatsley/models
FROM sheatsley/datasets
RUN pip install --no-cache-dir torch
COPY . models
RUN pip install --no-cache-dir models/ && rm -rf models
