import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, mean_squared_error, roc_auc_score
import joblib
import xgboost as xgb
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LogisticRegressionClassifier:
    def __init__(self, model_path):
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.model_path = model_path

    def train(self, X_train, y_train):
        print("Training logistic regression classifier model...")
        self.model.fit(X_train, y_train)
        train_pred_probs = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_pred_probs)
        print(f"Training complete. Training log loss: {train_loss}")

        return train_loss

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, label_encoder=None, evaluation_path=None):
        predictions = self.model.predict(X)
        prediction_probs = self.model.predict_proba(X)
        loss = log_loss(y, prediction_probs)
        acc = accuracy_score(y, predictions)
        auc_roc = roc_auc_score(y, prediction_probs, multi_class='ovr')  

        print(f"Log Loss: {loss}")
        print(f"Accuracy: {acc}")
        print(f"ROC AUC (multi-class): {auc_roc}")

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
            print(f"Confusion matrix saved to {cm_plot_path}")
        plt.close()

        return loss, acc, auc_roc

    def save_model(self):
        joblib.dump(self.model, self.model_path)

class XGBoostClassifier:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def train(self, X_train, y_train):
        self.model = xgb.XGBClassifier()  # Instantiate the model with default parameters
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
        
        # Calculate train loss
        train_pred_probs = self.model.predict_proba(X_train)
        train_loss = log_loss(y_train, train_pred_probs)
        print(f"Training complete. Training log loss: {train_loss}")
        
        return train_loss

    def bayesian_opt_tuning(self, X_train, y_train):
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
        print(f"Training with tuned parameters complete. Training log loss: {train_loss}")

        return train_loss

    def evaluate(self, X, y, evaluation_path=None):
        predictions = self.model.predict(X)
        prediction_probs = self.model.predict_proba(X)
        loss = log_loss(y, prediction_probs)
        acc = accuracy_score(y, predictions)
        auc_roc = roc_auc_score(y, prediction_probs, multi_class='ovr')  

        print(f"Log Loss: {loss}")
        print(f"Accuracy: {acc}")
        print(f"ROC AUC (multi-class): {auc_roc}")

        # Plotting feature importance
        plt.figure(figsize=(10, 7))
        xgb.plot_importance(self.model)
        plt.title("Feature Importance")
        plt.tight_layout()

        if evaluation_path:
            # Ensure the evaluation path directory exists
            os.makedirs(evaluation_path, exist_ok=True)
            # Append the filename to the path
            fi_plot_path = os.path.join(evaluation_path, 'feature_importance.png')
            plt.savefig(fi_plot_path)
            print(f"Feature importance plot saved to {fi_plot_path}")
        plt.close()

        return loss, acc, auc_roc

    def save_model(self):
        joblib.dump(self.model, self.model_path)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, num_classes) # Hidden layer to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)          # No activation needed for the output layer with CrossEntropyLoss
        return x
