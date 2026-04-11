#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# SageMaker passes the S3 config file to this directory
CONFIG_FILE="configs/configs.yaml"
OUTPUT_NOTEBOOK="notebooks/output/executed_notebook.ipynb"

echo "Starting Papermill execution..."
papermill /app/notebooks/train.ipynb \executed_notebook.ipynb \
    -p config_path \default.yaml

echo "Notebook execution completed successfully."