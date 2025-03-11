#!/bin/bash

# Name of the virtual environment
ENV_NAME="huggingface_venv"

# Python version to use
PYTHON_VERSION="python3.12"

# Directory to create the virtual environment
ENV_DIR="./${ENV_NAME}"

# Requirements file name
REQUIREMENTS_FILE="requirements.txt"

# Packages to install
TRANSFORMERS_VERSION="transformers"
TOKENIZERS_VERSION="tokenizers"
DATASETS_VERSION="datasets"

# Set default index URL for PyTorch packages if needed (e.g., for CPU)
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

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
echo "Upgrading pip..."
pip install --upgrade pip

# Install Hugging Face libraries
echo "Installing Hugging Face libraries..."
pip install ${TRANSFORMERS_VERSION} ${TOKENIZERS_VERSION} ${DATASETS_VERSION}

# Install PyTorch with CPU support (uncomment if needed)
# echo "Installing PyTorch with CPU support..."
# pip install torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}

# Optionally, create a requirements file
echo "Creating requirements file..."
pip freeze > ${REQUIREMENTS_FILE}
echo "Requirements file '${REQUIREMENTS_FILE}' created."

echo "Installation completed successfully."

# Provide instructions to activate the environment
echo "To activate the virtual environment, run: source ${ENV_DIR}/bin/activate"
