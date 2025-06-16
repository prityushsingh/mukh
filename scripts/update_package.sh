#!/bin/bash

# Activate conda environment
# Initialize conda
eval "$(conda shell.bash hook)"
conda activate mukh-dev

# Clean up old distribution files
rm -rf dist/*

# Build the package
if ! python -m build; then
    echo "Error: Package build failed"
    exit 1
fi

# Upload to PyPI
if ! python -m twine upload dist/*; then
    echo "Error: Package upload failed"
    exit 1
fi

echo "Package version $NEW_VERSION successfully built and uploaded"      