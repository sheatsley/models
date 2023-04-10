# Dockerfile for Deep Learning Models: https://github.com/sheatsley/models
FROM sheatsley/datasets
COPY . /models
RUN cd /models && pip install --no-cache-dir -e .
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
