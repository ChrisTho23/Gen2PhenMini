import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import h2o
from h2o.automl import H2OAutoML
import h2o.grid as hgrid
import joblib
import xgboost as xgb
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional

class LogisticRegressionClassifier:
    """
    Logistic Regression Classifier.

    Parameters:
    - model_path (str): The path to save the trained model.
    - class_weights (list or None): Weights associated with classes in the form [class_weight_1, class_weight_2, ...].
                                    If None, all classes are supposed to have weight one.

    Methods:
    - train(X_train, y_train): Train the logistic regression model.
    - predict(X): Predict the class labels for the input samples.
    - evaluate(X, y, label_encoder=None, evaluation_path=None): Evaluate the model's performance.
    - save_model(): Save the trained model to the specified path.
    """

    def __init__(self, model_path: str, class_weights: Optional[list] = None):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=class_weights)
        self.model_path = model_path

    def train(self, X_train, y_train):
        """
        Train the logistic regression model.

        Parameters:
        - X_train (array-like): The input training samples.
        - y_train (array-like): The target values.

        Returns:
        - train_loss (float): The log loss of the training data.
        """
        self.model.fit(X_train, y_train)
        train_pred_probs = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_pred_probs)

        return train_loss

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        - X (array-like): The input samples.

        Returns:
        - predictions (array-like): The predicted class labels.
        """
        return self.model.predict(X)

    def evaluate(self, X, y, label_encoder=None, evaluation_path=None):
        """
        Evaluate the model's performance.

        Parameters:
        - X (array-like): The input samples.
        - y (array-like): The target values.
        - label_encoder (object or None): The label encoder object to decode the labels for confusion matrix.
        - evaluation_path (str or None): The path to save the evaluation results.

        Returns:
        - loss (float): The log loss of the evaluation data.
        - acc (float): The accuracy score of the evaluation data.
        - auc_roc (float): The area under the ROC curve score of the evaluation data.
        """
        predictions = self.model.predict(X)
        prediction_probs = self.model.predict_proba(X)
        loss = log_loss(y, prediction_probs)
        acc = accuracy_score(y, predictions)
        auc_roc = roc_auc_score(y, prediction_probs, multi_class='ovr')  

        # Decode the labels for confusion matrix, if label_encoder is provided
        if label_encoder is not None:
            y_decoded = label_encoder.inverse_transform(y)
            predictions_decoded = label_encoder.inverse_transform(predictions)
            cm_labels = label_encoder.encoder.classes_
        else:
            y_decoded = y
            predictions_decoded = predictions
            cm_labels = sorted(np.unique(y_decoded).tolist())

        # Plotting confusion matrix
        cm = confusion_matrix(y_decoded, predictions_decoded, labels=cm_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_labels, yticklabels=cm_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if evaluation_path:
            # Ensure the evaluation path directory exists
            os.makedirs(evaluation_path, exist_ok=True)
            # Append the filename to the path
            cm_plot_path = os.path.join(evaluation_path, 'confusion_matrix.png')
            plt.savefig(cm_plot_path)
        plt.close()

        return loss, acc, auc_roc

    def save_model(self):
        """
        Save the trained model to the specified path.
        """
        joblib.dump(self.model, self.model_path)

class XGBoostClassifier:
    """
    XGBoostClassifier is a class that represents an XGBoost classifier model.

    Attributes:
        model (xgb.XGBClassifier): The XGBoost classifier model.
        model_path (str): The path to save/load the model.
        class_weights (list or None): The weights of each class for imbalanced classification.
    
    Methods:
        __init__(self, model_path: str, class_weights: Optional[list] = None, load_model: bool = False):
            Initializes the XGBoostClassifier object.
        
        load_model(self):
            Loads the XGBoost classifier model from the specified path.
        
        train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
            Trains the XGBoost classifier model on the given training data.
            Returns the training loss.
        
        bayesian_opt_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
            Performs Bayesian optimization to tune the hyperparameters of the XGBoost classifier model.
            Returns the training loss with tuned parameters.
        
        evaluate(self, X: np.ndarray, y: np.ndarray, label_encoder: Optional[LabelEncoder] = None,
                 evaluation_path: Optional[str] = None) -> Tuple[float, float, float]:
            Evaluates the XGBoost classifier model on the given data.
            Returns the loss, accuracy, and AUC-ROC score.
        
        save_model(self):
            Saves the XGBoost classifier model to the specified path.
    """
    def __init__(self, model_path: str, class_weights: Optional[list] = None, load_model: bool = False):
        """
        Initializes the XGBoostClassifier object.

        Args:
            model_path (str): The path to save/load the model.
            class_weights (list or None, optional): The weights of each class for imbalanced classification.
            load_model (bool, optional): Whether to load the model from the specified path.
        """
        self.model = None
        self.model_path = model_path
        self.class_weights = class_weights
        if load_model:
            self.load_model()

    def load_model(self):
        """
        Loads the XGBoost classifier model from the specified path.
        """
        self.model = joblib.load(self.model_path)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Trains the XGBoost classifier model on the given training data.

        Args:
            X_train (np.ndarray): The input features of the training data.
            y_train (np.ndarray): The target labels of the training data.

        Returns:
            float: The training loss.
        """
        if self.class_weights:
            self.model = xgb.XGBClassifier(scale_pos_weight=self.class_weights)
        else:
            self.model = xgb.XGBClassifier()
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
        
        # Calculate train loss
        train_pred_probs = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_pred_probs)
        
        return train_loss

    def bayesian_opt_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Performs Bayesian optimization to tune the hyperparameters of the XGBoost classifier model.

        Args:
            X_train (np.ndarray): The input features of the training data.
            y_train (np.ndarray): The target labels of the training data.

        Returns:
            float: The training loss with tuned parameters.
        """
        def xgb_evaluate(max_depth, gamma, colsample_bytree):
            params = {'eval_metric': 'logloss',
                      'max_depth': int(max_depth),
                      'gamma': gamma,
                      'colsample_bytree': colsample_bytree}
            cv_result = xgb.cv(params, xgb.DMatrix(X_train, label=y_train),
                               num_boost_round=100, nfold=3,
                               early_stopping_rounds=10,
                               metrics="logloss",
                               as_pandas=True)
            return -1.0 * cv_result['test-logloss-mean'].iloc[-1]

        xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10),
                                                     'gamma': (0, 1),
                                                     'colsample_bytree': (0.3, 0.9)})
        xgb_bo.maximize(init_points=2, n_iter=20)
        best_params = xgb_bo.max['params']
        best_params['max_depth'] = int(best_params['max_depth'])
        print(f"Best parameters found: {best_params}")

        # Set the model with the best parameters found from Bayesian Optimization
        self.model = xgb.XGBClassifier(**best_params)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)

        # Calculate train loss with tuned parameters
        train_pred_probs = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_pred_probs)

        return train_loss

    def evaluate(self, X: np.ndarray, y: np.ndarray, label_encoder: Optional[LabelEncoder] = None,
                 evaluation_path: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Evaluates the XGBoost classifier model on the given data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            label_encoder (LabelEncoder or None, optional): The label encoder for decoding the labels.
            evaluation_path (str or None, optional): The path to save the evaluation results.

        Returns:
            Tuple[float, float, float]: The loss, accuracy, and AUC-ROC score.
        """
        predictions = self.model.predict(X)
        prediction_probs = self.model.predict_proba(X)
        loss = log_loss(y, prediction_probs)
        acc = accuracy_score(y, predictions)
        auc_roc = roc_auc_score(y, prediction_probs, multi_class='ovr')  

        # Decode the labels for confusion matrix, if label_encoder is provided
        if label_encoder is not None:
            y_decoded = label_encoder.inverse_transform(y)
            predictions_decoded = label_encoder.inverse_transform(predictions)
            cm_labels = label_encoder.encoder.classes_
        else:
            y_decoded = y
            predictions_decoded = predictions
            cm_labels = sorted(np.unique(y_decoded).tolist())

        # Plotting confusion matrix
        cm = confusion_matrix(y_decoded, predictions_decoded, labels=cm_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_labels, yticklabels=cm_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title("Confusion Matrix")

        if evaluation_path:
            # Ensure the evaluation path directory exists
            os.makedirs(evaluation_path, exist_ok=True)
            # Append the filename to the path
            cm_plot_path = os.path.join(evaluation_path, 'confusion_matrix.png')
            plt.savefig(cm_plot_path)
        plt.close()

        # Plotting feature importance
        plt.figure(figsize=(10, 7))
        xgb.plot_importance(self.model)
        plt.title("Feature Importance")
        plt.tight_layout()

        if evaluation_path:
            # Append the filename to the path
            fi_plot_path = os.path.join(evaluation_path, 'feature_importance.png')
            plt.savefig(fi_plot_path)
        plt.close()

        return loss, acc, auc_roc

    def save_model(self):
        """
        Saves the XGBoost classifier model to the specified path.
        """
        joblib.dump(self.model, self.model_path)

class AutoMLClassifier:
    """
    AutoMLClassifier is a class that represents an AutoML classifier model.

    Attributes:
        model (H2OAutoML): The AutoML model.
        model_path (str): The path to save/load the model.
    
    Methods:
        __init__(self, model_path: str, max_models: int = 20, max_runtime_secs: Optional[int] = None,
                 class_weights: bool = False):
            Initializes the AutoMLClassifier object.
        
        train(self, X_train, y_train):
            Trains the AutoML classifier model on the given training data.
        
        evaluate(self, X_test, y_test) -> Tuple[float, float, float]:
            Evaluates the AutoML classifier model on the given data.
            Returns the loss, accuracy, and AUC-ROC score.
        
        save_model(self):
            Saves the AutoML classifier model to the specified path.
    """
    def __init__(self, model_path: str, max_models: int = 20, max_runtime_secs: Optional[int] = None,
                 class_weights: bool = False):
        """
        Initializes the AutoMLClassifier object.

        Args:
            model_path (str): The path to save/load the model.
            max_models (int): The maximum number of models to train in the AutoML process. Default is 20.
            max_runtime_secs (int or None): The maximum runtime in seconds for the AutoML process. Default is None.
            class_weights (bool): Whether to balance the classes by assigning weights. Default is False.
        """
        h2o.init()
        if class_weights:
            self.model = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1, balance_classes=True)
        else:
            self.model = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1)
        self.model_path = model_path

    def train(self, X_train, y_train):
        """
        Trains the AutoML classifier model on the given training data.

        Args:
            X_train: The input features for training.
            y_train: The target labels for training.
        """
        # Convert to H2OFrame
        train_frame = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        feature_names = list(X_train.columns)
        target_name = y_train.name

        # Set target and predictor variables
        train_frame[target_name] = train_frame[target_name].asfactor()

        # Train the model
        self.model.train(x=feature_names, y=target_name, training_frame=train_frame)

    def evaluate(self, X_test, y_test) -> Tuple[float, float, float]:
        """
        Evaluates the AutoML classifier model on the given data.

        Args:
            X_test: The input features for testing.
            y_test: The target labels for testing.

        Returns:
            Tuple[float, float, float]: The loss, accuracy, and AUC-ROC score.
        """
        test_frame = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
        target_name = y_test.name

        # Make predictions
        preds = self.model.predict(test_frame).as_data_frame()
        # H2O AutoML predictions include class probabilities and class labels. Choose what you need:
        predictions = preds["predict"].values
        pred_probs = preds.drop(columns=["predict"]).values

        # Evaluate performance
        loss = log_loss(y_test, pred_probs)
        acc = accuracy_score(y_test, predictions)
        auc_roc = roc_auc_score(y_test, pred_probs, multi_class='ovr')

        return loss, acc, auc_roc

    def save_model(self):
        """
        Saves the AutoML classifier model to the specified path.
        """
        # Save the model
        joblib.dump(self.model, self.model_path)

class SimpleNN(nn.Module):
    """
    SimpleNN is a class that represents a simple neural network model.

    Attributes:
        fc1 (nn.Linear): The linear layer from input to hidden layer.
        fc2 (nn.Linear): The linear layer from hidden layer to output layer.

    Methods:
        __init__(self, input_size: int, hidden_size: int, num_classes: int):
            Initializes the SimpleNN object.
        
        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs forward pass through the neural network.
            Returns the output tensor.
        
        save_model(self, filename: str):
            Saves the model's state dictionary to the specified file.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, num_classes) # Hidden layer to output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)          # No activation needed for the output layer with CrossEntropyLoss
        return x

    def save_model(self, filename: str):
        """
        Saves the model's state dictionary to the specified file.

        Args:
            filename (str): The path to save the model's state dictionary.
        """
        torch.save(self.state_dict(), filename)
