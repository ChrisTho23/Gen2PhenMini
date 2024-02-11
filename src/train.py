import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Union, Any, Optional

from config import DATA, MODEL, ENCODER, EVALUATION
from preprocessing import LabelEncoderWrapper
from models import LogisticRegressionClassifier, XGBoostClassifier, AutoMLClassifier, SimpleNN

def test_train_split(df: pd.DataFrame, target_column: str = 'eye_color', weight: bool = False) -> Union[tuple, tuple, Dict[str, float]]:
    """
    Split the data into train and test sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The column name of the target variable. Default is 'eye_color'.
        weight (bool): Flag to compute class weights. Default is False.

    Returns:
        If weight is False:
            tuple: A tuple containing X_train, X_test, y_train, y_test.
        If weight is True:
            tuple: A tuple containing X_train, X_test, y_train, y_test, class_weights_dict.
    """
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

def train_linear_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, metrics: Dict[str, Dict[str, float]], model_name: str, label_encoder: Optional[LabelEncoder] = None, class_weights_dict: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """
    Trains a linear model using logistic regression.

    Args:
        X_train (np.ndarray): The training data features.
        y_train (np.ndarray): The training data labels.
        X_test (np.ndarray): The testing data features.
        y_test (np.ndarray): The testing data labels.
        metrics (Dict[str, Dict[str, float]]): A dictionary to store the evaluation metrics.
        model_name (str): The name of the model.
        label_encoder (Optional[LabelEncoder], optional): The label encoder. Defaults to None.
        class_weights_dict (Optional[Dict[str, float]], optional): The class weights dictionary. Defaults to None.

    Returns:
        Dict[str, Dict[str, float]]: The updated metrics dictionary.
    """
    print(f"Train {model_name} model...")
    if class_weights_dict is not None:
        linear_classifier = LogisticRegressionClassifier(MODEL[model_name])
    else:
        linear_classifier = LogisticRegressionClassifier(MODEL[model_name], class_weights=class_weights_dict)
    linear_train_loss = linear_classifier.train(X_train, y_train)
    linear_test_loss, linear_test_acc, linear_test_aucroc =  linear_classifier.evaluate(
        X_test, y_test, label_encoder=label_encoder, evaluation_path=EVALUATION[model_name]
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

def train_xgb_model(X_train: Any, y_train: Any, X_test: Any, y_test: Any, model_name: str, metrics: Dict[str, Dict[str, Any]], label_encoder: Optional[Any] = None, class_weights_dict: Optional[Dict[Any, Any]] = None, fine_tuning: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Trains an XGBoost model using the given training and testing data.

    Args:
        X_train (Any): The training data.
        y_train (Any): The training labels.
        X_test (Any): The testing data.
        y_test (Any): The testing labels.
        model_name (str): The name of the model.
        metrics (Dict[str, Dict[str, Any]]): A dictionary to store the evaluation metrics.
        label_encoder (Optional[Any], optional): The label encoder. Defaults to None.
        class_weights_dict (Optional[Dict[Any, Any]], optional): The class weights dictionary. Defaults to None.
        fine_tuning (bool, optional): Whether to perform fine-tuning. Defaults to False.

    Returns:
        Dict[str, Dict[str, Any]]: The updated metrics dictionary.
    """
    print(f"Fine tuning {model_name} model...")
    if class_weights_dict is not None:
        xgb_classifier = XGBoostClassifier(MODEL[model_name], class_weights=class_weights_dict)
    else:
        xgb_classifier = XGBoostClassifier(MODEL[model_name])

    if fine_tuning:
        xgb_train_loss = xgb_classifier.bayesian_opt_tuning(X_train, y_train)
    else:
        xgb_train_loss = xgb_classifier.train(X_train, y_train)

    xgb_test_loss, xgb_test_acc, xgb_test_aucroc = xgb_classifier.evaluate(X_test, y_test, label_encoder=label_encoder, evaluation_path=EVALUATION[model_name])
    xgb_classifier.save_model()

    # Logging the metrics
    metrics[model_name] = {
        'train_loss': xgb_train_loss,
        'test_loss': xgb_test_loss,
        'test_acc': xgb_test_acc,
        'test_aucroc': xgb_test_aucroc,
    }

    return metrics

def train_automl_model(X_train: Any, y_train: Any, X_test: Any, y_test: Any, model_name: str, metrics: Dict[str, Dict[str, float]], class_weights: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Trains an AutoML model using the given training and testing data.

    Args:
        X_train (Any): The training data features.
        y_train (Any): The training data labels.
        X_test (Any): The testing data features.
        y_test (Any): The testing data labels.
        model_name (str): The name of the model.
        metrics (Dict[str, Dict[str, float]]): A dictionary to store the metrics.
        class_weights (bool, optional): Whether to use class weights during training. Defaults to False.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the metrics of the trained model.
    """, 
    automl = AutoMLClassifier(model_path=MODEL[model_name], class_weights=class_weights)
    automl_train_loss = automl.train(X_train, y_train)
    automl_test_loss, automl_test_acc, automl_test_aucroc = automl.evaluate(X_test, y_test)
    automl.save_model()

    # Logging the metrics
    metrics[model_name] = {
        'train_loss': automl_train_loss,
        'test_loss': automl_test_loss,
        'test_acc': automl_test_acc,
        'test_aucroc': automl_test_aucroc,
    }

    return metrics

def train_neural_network(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, 
                         input_size: int, hidden_size: int, num_classes: int, 
                         learning_rate: float = 0.001, batch_size: int = 32, num_epochs: int = 30, 
                         metrics: Optional[Dict[str, Dict[str, Union[float, str]]]] = None, 
                         model_name: Optional[str] = None, 
                         class_weights_dict: Optional[Dict[int, float]] = None) -> Optional[Dict[str, Dict[str, Union[float, str]]]]:
    """
    Trains a neural network model using the given training and testing data.

    Args:
        X_train (pd.DataFrame): The input features of the training data.
        y_train (pd.Series): The target labels of the training data.
        X_test (pd.DataFrame): The input features of the testing data.
        y_test (pd.Series): The target labels of the testing data.
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layer.
        num_classes (int): The number of classes in the target labels.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        num_epochs (int, optional): The number of epochs for training. Defaults to 30.
        metrics (Optional[Dict[str, Dict[str, Union[float, str]]]], optional): A dictionary to store the metrics. 
            Defaults to None.
        model_name (Optional[str], optional): The name of the model. Defaults to None.
        class_weights_dict (Optional[Dict[int, float]], optional): A dictionary of class weights. 
            Defaults to None.

    Returns:
        Optional[Dict[str, Dict[str, Union[float, str]]]]: A dictionary containing the metrics of the trained model.
    """
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
    # Apply class weights if provided
    if class_weights_dict is not None:
        class_weights = torch.tensor([class_weights_dict.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
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
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR_NO_NAN', label_encoder=eyecolor_encoder)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_NO_NAN', metrics, label_encoder=eyecolor_encoder)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE_NO_NAN', metrics, label_encoder=eyecolor_encoder, fine_tuning=True)
    metrics = train_automl_model(X_train, y_train, X_test, y_test, 'AUTOML_NO_NAN', metrics)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=6, hidden_size=10, num_classes=7,
                                    learning_rate=0.001, batch_size=32, num_epochs=30,
                                    metrics=metrics, model_name="NEURAL_NETWORK_NO_NAN")

    print("Train models with class weights...")
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR_NO_NAN_WEIGHTED', label_encoder=eyecolor_encoder, class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_NO_NAN_WEIGHTED', metrics, label_encoder=eyecolor_encoder, class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE_NO_NAN_WEIGHTED', metrics, label_encoder=eyecolor_encoder, fine_tuning=True, class_weights_dict=class_weights_dict)
    metrics = train_automl_model(X_train, y_train, X_test, y_test, 'AUTOML_NO_NAN_WEIGHTED', metrics, class_weights=True)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=6, hidden_size=10, num_classes=7,
                                learning_rate=0.001, batch_size=32, num_epochs=30,
                                metrics=metrics, model_name="NEURAL_NETWORK_NO_NAN_WEIGHTED", class_weights_dict=class_weights_dict)

    print(f"Training models on data with encoded NaN values...")

    df = pd.read_csv(DATA['FINAL'])

    X_train, X_test, y_train, y_test, class_weights_dict = test_train_split(df, weight=True)
    
    print("Train models without class weights...")
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR', label_encoder=eyecolor_encoder)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST', metrics, label_encoder=eyecolor_encoder)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE', metrics, label_encoder=eyecolor_encoder, fine_tuning=True)
    metrics = train_automl_model(X_train, y_train, X_test, y_test, 'AUTOML', metrics)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=13, hidden_size=10, num_classes=7,
                                        learning_rate=0.001, batch_size=32, num_epochs=30,
                                        metrics=metrics, model_name="NEURAL_NETWORK")

    print("Train models with class weights...")
    metrics = train_linear_model(X_train, y_train, X_test, y_test, metrics, 'LINEAR_WEIGHTED', label_encoder=eyecolor_encoder, class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_WEIGHTED', metrics, label_encoder=eyecolor_encoder, class_weights_dict=class_weights_dict)
    metrics = train_xgb_model(X_train, y_train, X_test, y_test, 'XGBOOST_FINE_WEIGHTED', metrics, label_encoder=eyecolor_encoder, fine_tuning=True, class_weights_dict=class_weights_dict)
    metrics = train_automl_model(X_train, y_train, X_test, y_test, 'AUTOML_WEIGHTED', metrics, class_weights=True)
    metrics = train_neural_network(X_train, y_train, X_test, y_test, input_size=13, hidden_size=10, num_classes=7,
                                learning_rate=0.001, batch_size=32, num_epochs=30,
                                metrics=metrics, model_name="NEURAL_NETWORK_WEIGHTED", class_weights_dict=class_weights_dict)

    with open(EVALUATION["METRICS"], 'w') as file:
        json.dump(metrics, file, indent=4)
    

