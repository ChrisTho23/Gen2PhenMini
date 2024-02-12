# Gen2PhenMini: Eye Color Prediction from SNP Data

## Overview
This repository contains a comprehensive analysis and machine learning approach for predicting eye color from SNP (Single Nucleotide Polymorphisms) data. The project utilizes various data processing techniques, class balancing methods, and multiple machine learning models to address the challenge of eye color prediction, a trait with high heritability.

## Management summary
The project trained a fine-tuned XGBoost classifier able to predict the eye-color of an individual with an accuracy of ca. 54% based on the genotype information. 
- All the data used in this project is from the open-source plattform OpenSNP (https://opensnp.org).
- Eyecolor has been selected as target phenotype due data abundance (1800 users indicated eyecolor variation) and heritability (ca. 80%).
- Determined 8 SNPs (rs12896399 at SLC24A4, rs12913832 at HERC2, rs1545397 at OCA2, rs16891982 at SLC45A2, rs1426654 at SLC24A5, rs885479 at MC1R, rs6119471 at ASIP, and rs12203592 at IRF4) that influence eyecolor of an individual and their respective reference allele.
- Fetched data from all users that indicated their eyecolor and genetic information on at least 1 of the 8 SNPs from OpenSNP.
- Treated NaN values and target class imbalance (comparison of removing NaN vs. encoding NaNs and including class weighting vs. not including class weighting led to the final configation: remove NaN and not balance classes).
- Encoded genetic information in ternary BCF system based on reference allele
- Trained and compared several machine learning models: Logistic Regression, XGBoost, AutoML, and a simple neural network. Two-layer neural network and fine-tuned XGBoost classsifier perform best at around 54% accuracy.
- Explored confusion matrix and feature importance of fine-tuned XGBoost model to identify further ways to improve model performance. We learn that the model is good at differentiating between distince variation such as green and blue eyes but lacks data to accurutely set apart more similar eyecolors such as brown and hazel or blue and blue-grey.

## To go further
- Feature engineering: It could be beneficial for the model to add some more features such as: (1) Which allele matches reference? Maternal or Paternal allele? (2) Sex of the person.
- Gather more data: Eyecolor is a polygenic gene. That means not only one but the combination of many loci is repsonsible for the final eye-color variation. Thus, collecting extensive data for all relelvant SNP is expected to increase model performance. Especially since SNPs other than the 'key identifier' may help us to distinct more subtle differences in eyecolor such as between Brown and Hazel. Morevover, having more data samples in general will also help the model to generalize better for underrepresented eyecolors as there is only very little data available for these classes to date.
- Increase data quality: As it is often the case in open source datasets, the cluster is of mediocre quality which hampens model performance. For instance, the target, eye color, is described very inconsistently which can lead to missclassification.
- Depending on if the classification model's application puts more emphasis on explainability/transparency or accuracy, put the fine-tuned XGBoost model or a fine-tuned version of the nueral network into production

## Requirements
The required libraries and dependencies are listed in `requirements.txt`. This includes packages like pandas, numpy, scikit-learn, and others essential for data processing and machine learning.

## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. The [helper notebook](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/helper.ipynb) gives a exhaustive summary of the repository, read it to get an idea of what is going on.
4. Run the [data script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/data.py) to get all relevant data from the OpenSNP website.
5. Run the [preprocessing script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/preprocessing.py) preprocess the data.
6. Run the [train script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/train.py) to train all the models defined in the [models script](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/src/models.py). You can find relevant metrics to evaluate the model performance in the [evaluation folder](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/evaluation). The trained models can be retrieved from the [models folder](https://github.com/ChrisTho23/Gen2PhenMini/blob/main/models)