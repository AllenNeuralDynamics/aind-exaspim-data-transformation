FROM python:3.12-slim
LABEL maintainer="Carson Berry <carson.berry@alleninstitute.org>"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY pyproject.toml README.md setup.py ./
COPY src ./src

# Install dependencies
RUN pip install --no-cache-dir pip setuptools setuptools-scm && \
    pip install --no-cache-dir .

# Set the entrypoint for the container
ENTRYPOINT ["/bin/bash"]
