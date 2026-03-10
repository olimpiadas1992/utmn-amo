#!/bin/bash

echo "Step 1: Creating data"
python scripts/data_creation.py

echo "Step 2: Preprocessing"
python scripts/model_preprocessing.py

echo "Step 3: Training model"
python scripts/model_preparation.py

echo "Step 4: Testing model"
python scripts/model_testing.py

echo "Pipeline finished"