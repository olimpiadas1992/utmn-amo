#!/bin/bash

echo "Step 1: Creating data"
python data_creation.py

echo "Step 2: Preprocessing"
python model_preprocessing.py

echo "Step 3: Training model"
python model_preparation.py

echo "Step 4: Testing model"
python model_testing.py

echo "Pipeline finished"