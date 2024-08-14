#!/bin/bash

# Update package list and install the venv package (uncomment if necessary)
#sudo apt-get update
#sudo apt-get install python3-venv  # or just python3-venv if using the default python version

# Name of the virtual environment
ENV_NAME="huggingface_venv"

# Python version to use
PYTHON_VERSION="python3.12"

# Directory to create the virtual environment
ENV_DIR="./${ENV_NAME}"

# Requirements file name
REQUIREMENTS_FILE="requirements.txt"

# Hugging Face Transformers version
TRANSFORMERS_VERSION="transformers"

# Tokenizers version
TOKENIZERS_VERSION="tokenizers"

# Datasets version
DATASETS_VERSION="datasets"

echo "Setting up a virtual environment for Hugging Face..."

# Create virtual environment using venv
if ! ${PYTHON_VERSION} -m venv ${ENV_DIR}; then
    echo "Failed to create a virtual environment!"
    exit 1
fi

echo "Virtual environment '${ENV_NAME}' created."

# Activate the virtual environment
source ${ENV_DIR}/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Hugging Face libraries
pip install ${TRANSFORMERS_VERSION} ${TOKENIZERS_VERSION} ${DATASETS_VERSION}

# Optionally, you can create a requirements file
pip freeze > ${REQUIREMENTS_FILE}
echo "Requirements file '${REQUIREMENTS_FILE}' created."

echo "Installation completed successfully."

# Provide instructions to activate the environment
echo "To activate the virtual environment, run: source ${ENV_DIR}/bin/activate"