import pandas as pd
import joblib

from models import XGBoostClassifier
from preprocessing import LabelEncoderWrapper
from train import test_train_split
from config import DATA, MODEL, EVALUATION, ENCODER

if __name__ == "__main__":
    print("Evaluate fine-tuned XGBoost model")
    model = XGBoostClassifier(model_path=MODEL['XGBOOST_FINE_NO_NAN'], load_model=True)
    eyecolor_encoder = joblib.load(ENCODER['EYECOLOR'])

    df_nonan = pd.read_csv(DATA['FINAL_NO_NAN'])
    X_train, X_test, y_train, y_test = test_train_split(df_nonan)

    _, _, _ = model.evaluate(X_test, y_test, label_encoder=eyecolor_encoder, evaluation_path=EVALUATION['XGBOOST_FINE_NO_NAN'])
