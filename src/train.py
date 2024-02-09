import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATA, MODEL, ENCODER, EVALUATION
from preprocessing import LabelEncoderWrapper
from models import LogisticRegressionClassifier, XGBoostClassifier, SimpleNN

def test_train_split(df, target_column='eye_color'):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    eyecolor_encoder = joblib.load(ENCODER['EYECOLOR'])
    df_nonan = pd.read_csv(DATA['PREPROCESSED_NO_NAN'])
    
    metrics = {}

    print("Training logistic regression on dataset without NaN...")

    X_train, X_test, y_train, y_test = test_train_split(df_nonan)

    linear_classifier_no_nan = LogisticRegressionClassifier(MODEL['LINEAR_NO_NAN'])
    linear_no_nan_train_loss = linear_classifier_no_nan.train(X_train, y_train)
    linear_no_nan_test_loss, linear_no_nan_test_acc, linear_no_nan_test_aucroc =  linear_classifier_no_nan.evaluate(
        X_test, y_test, eyecolor_encoder, evaluation_path=EVALUATION['LINEAR_NO_NAN']
    )
    linear_classifier_no_nan.save_model()

    # Logging the metrics
    metrics['LINEAR_NO_NAN'] = {
        'train_loss': linear_no_nan_train_loss,
        'test_loss': linear_no_nan_test_loss,
        'test_acc': linear_no_nan_test_acc,
        'test_aucroc': linear_no_nan_test_aucroc,
    }

    df = pd.read_csv(DATA['PREPROCESSED'])

    print("Training logistic regression on dataset with encoded NaN...")

    X_train, X_test, y_train, y_test = test_train_split(df)

    linear_classifier = LogisticRegressionClassifier(MODEL['LINEAR'])
    linear_train_loss = linear_classifier.train(X_train, y_train)
    linear_test_loss, linear_test_acc, linear_test_aucroc =  linear_classifier.evaluate(
        X_test, y_test, eyecolor_encoder, evaluation_path=EVALUATION['LINEAR']
    )
    linear_classifier.save_model()

    # Logging the metrics
    metrics['LINEAR_ENCODED'] = {
        'train_loss': linear_train_loss,
        'test_loss': linear_test_loss,
        'test_acc': linear_test_acc,
        'test_aucroc': linear_test_aucroc,
    }

    print("Training XGBoost model...")

    X_train, X_test, y_train, y_test = test_train_split(df)

    xgb_classifier = XGBoostClassifier(MODEL['XGBOOST'])
    xgb_train_loss = xgb_classifier.train(X_train, y_train)
    xgb_test_loss, xgb_test_acc, xgb_test_aucroc = xgb_classifier.evaluate(X_test, y_test, evaluation_path=EVALUATION['XGBOOST'])
    xgb_classifier.save_model()

    # Logging the metrics
    metrics['XGBOOST'] = {
        'train_loss': xgb_train_loss,
        'test_loss': xgb_test_loss,
        'test_acc': xgb_test_acc,
        'test_aucroc': xgb_test_aucroc,
    }

    print("Training fine-tuned XGBoost model...")

    xgb_classifier_fine = XGBoostClassifier(MODEL['XGBOOST_FINE'])
    xgb_fine_train_loss = xgb_classifier_fine.bayesian_opt_tuning(X_train, y_train)
    xgb_fine_test_loss, xgb_fine_test_acc, xgb_fine_test_aucroc = xgb_classifier_fine.evaluate(X_test, y_test, evaluation_path=EVALUATION['XGBOOST_FINE'])
    xgb_classifier_fine.save_model()

    # Logging the metrics
    metrics['XGBOOST_FINE'] = {
        'train_loss': xgb_fine_train_loss,
        'test_loss': xgb_fine_test_loss,
        'test_acc': xgb_fine_test_acc,
        'test_aucroc': xgb_fine_test_aucroc,
    }

    print("Training neural network...")

    input_size = 6
    hidden_size = 10
    num_classes = 8

    # Convert dataframe to numpy array and then to PyTorch tensor
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train = torch.tensor(y_train.values, dtype=torch.int64)
    X_val = torch.tensor(X_test.values, dtype=torch.float32)
    Y_val = torch.tensor(y_test.values, dtype=torch.int64)

    # Create Dataset objects
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    # Create DataLoader objects
    batch_size = 32  # Define your batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Define your model
    input_size = 6  # Number of features
    hidden_size = 50  
    num_classes = 8  # Number of classes
    model = SimpleNN(input_size, hidden_size, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 30  

    # Training loop
    for epoch in range(num_epochs):
        model.train() 
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validation loop (after training)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    with open(EVALUATION["METRICS"], 'w') as file:
        json.dump(metrics, file, indent=4)
    

