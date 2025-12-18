#!/bin/bash
# setup.sh

set -e  # Exit on any error

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create --name clm python=3.10 pip -y
conda activate clm

# Install main requirements
conda env update --file environment.yml

# Install s4dd from source
if [ ! -d "s4-for-de-novo-drug-design" ]; then
    git clone https://github.com/molML/s4-for-de-novo-drug-design.git
fi
cd s4-for-de-novo-drug-design
pip install -e .
cd ..

# Install CLM package
pip install -e . --no-deps

echo "Environment setup complete!"