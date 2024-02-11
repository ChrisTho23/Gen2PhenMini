# Gen2PhenMini: Eye Color Prediction from SNP Data

## Overview
This repository contains a comprehensive analysis and machine learning approach for predicting eye color from SNP (Single Nucleotide Polymorphisms) data. The project utilizes various data processing techniques, class balancing methods, and multiple machine learning models to address the challenge of eye color prediction, a trait with high heritability.

## Features
- Utilizes 8 selected SNPs known to be associated with eye color.
- Implements data preprocessing, including handling of missing values and one-hot encoding.
- Addresses class imbalance in eye color categories using computed class weights.
- Employs several machine learning models: Logistic Regression, XGBoost, AutoML, and a simple neural network.
- Includes model evaluation and comparison to identify the best performing approach.
- Focus on explainability and interpretability of the model results, particularly for best-performing model (fine-tuned XGBoost model).

## Dataset from OpenSNP (https://opensnp.org)
- The dataset comprises SNP data, annotations, and phenotype information focused on eye color.
- The SNP data is encoded using a ternary system based on reference allele and preprocessed for analysis.
- The final dataset consists of both versions with encoded NaN values and without NaN values.

## Requirements
The required libraries and dependencies are listed in `requirements.txt`. This includes packages like pandas, numpy, scikit-learn, and others essential for data processing and machine learning.

## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. The [helper notebook](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/helper.ipynb) gives a exhaustive summary of the repository, read it to get an idead of what is going on.
4. Run the [data script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/data.py) to get all relevant data from the OpenSNP website.
5. Run the [preprocessing script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/preprocessing.py) preprocess the data.
6. Run the [train script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/train.py) to train all the models defined in the [models script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/models.py). You can find relevant metrics to evaluate the model performance in the [evaluation folder](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/evaluation). The trained models can be retrieved from the [models folder](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/models)

## Future Work
- Enhancing feature engineering by considering additional genetic and non-genetic factors.
- Expanding the dataset to include more relevant SNPs and increase the sample size, particularly for underrepresented eye colors.
- Improving data quality, especially in terms of the consistency of eye color descriptions.