# Base Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY . /workspace/
RUN pip install -r requirements.txt
RUN pip install -e .
RUN pip install jupyterlab
