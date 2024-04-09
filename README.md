# PD assignment

## Overview

This Python script processes neuroimaging data, performing analysis and classification of patient diagnoses based on imaging features. It uses libraries such as nibabel for handling NIfTI files, pandas and numpy for data manipulation, matplotlib for visualization, and scikit-learn for machine learning tasks.

## Requirements

Ensure you have Python installed on your system. Required packages can be found in the requirements.txt file. Install them using:

pip install -r requirements.txt

## Data Structure
The script expects NIfTI files (*.nii) for volume of interest (VOI) templates and measurement data in specific directorie: data. 

## Features

    Load and process neuroimaging data in NIfTI format.
    Calculate mean measurements for specified brain structures.
    Perform diagnosis classification using logistic regression.
    Evaluate the model using cross-validation and display accuracy metrics.

## Usage

    Place your NIfTI files in the appropriate directorie.
    Modify paths in the script if necessary to match your directory structure.
    Run the script in a Python environment:

python train_model.py

    Check the output in the console and any generated files for predictions and visualizations.

## Output

    The script prints mean measurements for brain structures, cross-validation accuracy, and model coefficients to the console.
    Generates histograms for model accuracies and prediction distributions.
    Saves a CSV file with diagnosis predictions for test data.

## Customization

To analyze different structures or use other machine learning models, adjust the labels_structures dictionaries and the model in the script.