import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATA, MODEL, ENCODER, EVALUATION
from preprocessing import LabelEncoderWrapper
from models import LogisticRegressionClassifier, XGBoostClassifier, AutoMLClassifier, SimpleNN

def test_train_split(df, target_column='eye_color', weight=False):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Compute class weights if weight flag is set
    if weight:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        return X_train, X_test, y_train, y_test, class_weights_dict
    
    return X_train, X_test, y_train, y_test

def train_linear_model(X_train, y_train, X_test, y_test, metrics, model_name, class_weights_dict=None):
    print(f"Train {model_name} model...")
    if class_weights_dict is None:
        linear_classifier = LogisticRegressionClassifier(MODEL[model_name])
    else:
        linear_classifier = LogisticRegressionClassifier(MODEL[model_name], class_weights=class_weights_dict)
    linear_train_loss = linear_classifier.train(X_train, y_train)
    linear_test_loss, linear_test_acc, linear_test_aucroc =  linear_classifier.evaluate(
        X_test, y_test, eyecolor_encoder, evaluation_path=EVALUATION[model_name]
    )
    linear_classifier.save_model()

    print(f"Finished training {model_name} model...")
    print(f"Test accuracy: {linear_test_acc:.2f}")

    # Logging the metrics
    metrics[model_name] = {
        'train_loss': linear_train_loss,
        'test_loss': linear_test_loss,
        'test_acc': linear_test_acc,
        'test_aucroc': linear_test_aucroc,
    }

    return metrics

def train_xgb_model(X_train, y_train, X_test, y_test, model_name, metrics, class_weights_dict=None, fine_tuning=False):
    print(f"Fine tuning {model_name} model...")
    if class_weights_dict:
        xgb_classifier = XGBoostClassifier(MODEL[model_name], class_weights=class_weights_dict)
    else:
        xgb_classifier = XGBoostClassifier(MODEL[model_name])

    if fine_tuning:
        xgb_train_loss = xgb_classifier.bayesian_opt_tuning(X_train, y_train)
    else:
        xgb_train_loss = xgb_classifier.train(X_train, y_train)

    xgb_test_loss, xgb_test_acc, xgb_test_aucroc = xgb_classifier.evaluate(X_test, y_test, evaluation_path=EVALUATION[model_name])
    xgb_classifier.save_model()

    # Logging the metrics
    metrics[model_name] = {
        'train_loss': xgb_train_loss,
        'test_loss': xgb_test_loss,
        'test_acc': xgb_test_acc,
        'test_aucroc': xgb_test_aucroc,
    }

    return metrics

def train_automl_model(X_train, y_train, X_test, y_test, model_name, metrics):
    automl = AutoMLClassifier(MODEL[model_name])
    automl_train_loss = automl.train(X_train, y_train)
    automl_test_loss, automl_test_acc, automl_test_aucroc = automl.evaluate(X_test, y_test, evaluation_path=EVALUATION[model_name])
    automl.save_model()
    automl.print_selected_models()

    # Logging the metrics
    metrics[model_name] = {
        'train_loss': automl_train_loss,
        'test_loss': automl_test_loss,
        'test_acc': automl_test_acc,
        'test_aucroc': automl_test_aucroc,
    }

    return metrics

def train_neural_network(X_train, y_train, X_test, y_test, input_size, hidden_size, num_classes, 
                         learning_rate=0.001, batch_size=32, num_epochs=30, 
                         metrics=None, model_name="neural_network_model"):
    print(f"Train {model_name} model...")
    # Data preparation
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train = torch.tensor(y_train.values, dtype=torch.int64)
    X_val = torch.tensor(X_test.values, dtype=torch.float32)
    Y_val = torch.tensor(y_test.values, dtype=torch.int64)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    for epoch in range(num_epochs):
        model.train() 
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validation and calculating metrics
    model.eval()
    val_losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    val_accuracy = np.mean(np.array(y_true) == np.array(y_pred))

    # Update metrics dictionary
    if metrics is not None:
        metrics[model_name] = {
            'train_loss': np.mean(train_losses),
            'test_loss': np.mean(val_losses),
            'test_acc': val_accuracy,
            'test_aucroc': 'Not available',
        }

    # Save the model
    model.save_model(MODEL[model_name])

    return metrics


if __name__ == "__main__":
    eyecolor_encoder = joblib.load(ENCODER['EYECOLOR'])
    metrics = {}

    print(f"Training models on data without NaN values...")

    df_nonan = pd.read_csv(DATA['FINAL_NO_NAN'])

    X_train, X_test, y_train, y_test, class_weights_dict = test_train_split(df_nonan, weight=True)

    print("Train models without class weights...")
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR_NO_NAN')
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_NO_NAN', metrics)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE_NO_NAN', metrics, fine_tuning=True)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=6, hidden_size=10, num_classes=7,
                                    learning_rate=0.001, batch_size=32, num_epochs=30,
                                    metrics=metrics, model_name="NEURAL_NETWORK_NO_NAN")

    print("Train models with class weights...")
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR_NO_NAN_WEIGHTED', class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_NO_NAN_WEIGHTED', metrics, class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE_NO_NAN_WEIGHTED', metrics, fine_tuning=True, class_weights_dict=class_weights_dict)

    print(f"Training models on data with encoded NaN values...")

    df = pd.read_csv(DATA['FINAL'])

    X_train, X_test, y_train, y_test, class_weights_dict = test_train_split(df, weight=True)

    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR')
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST', metrics)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE', metrics, fine_tuning=True)
    #metrics = train_automl_model(X_train, y_train, X_test, y_test, 'AUTOML', metrics)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=13, hidden_size=10, num_classes=7,
                                        learning_rate=0.001, batch_size=32, num_epochs=30,
                                        metrics=metrics, model_name="NEURAL_NETWORK")

    with open(EVALUATION["METRICS"], 'w') as file:
        json.dump(metrics, file, indent=4)
    

