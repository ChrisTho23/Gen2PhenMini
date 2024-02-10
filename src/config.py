from pathlib import Path

DATA = {
    'GEN': Path('../data/gen.csv'),
    'PHEN': Path('../data/phen.csv'),
    'USER': Path('../data/user.csv'),
    'ANNOTATIONS': Path('../data/annotations.csv'),
    'CLEAN': Path('../data/clean.csv'),
    'ENCODED': Path('../data/encoded.csv'),
    'PREPROCESSED': Path('../data/preprocessed.csv'),
    'PREPROCESSED_NO_NAN': Path('../data/preprocessed_no_nan.csv'),
    'FINAL': Path('../data/final.csv'),
    'FINAL_NO_NAN': Path('../data/final_no_nan.csv'),
}

MODEL = {
    'LINEAR': Path('../models/logistic_regression.pkl'),
    'LINEAR_NO_NAN': Path('../models/logistic_regression_no_nan.pkl'),
    'LINEAR_NO_NAN_WEIGHTED': Path('../models/logistic_regression_no_nan_weighted.pkl'),
    'XGBOOST': Path('../models/xgboost.pkl'),
    'XGBOOST_NO_NAN': Path('../models/xgboost_no_nan.pkl'),
    'XGBOOST_NO_NAN_WEIGHTED': Path('../models/xgboost_no_nan_weighted.pkl'),
    'XGBOOST_FINE': Path('../models/xgboost_fine.pkl'),
    'XGBOOST_FINE_NO_NAN': Path('../models/xgboost_fine_no_nan.pkl'),
    'XGBOOST_FINE_NO_NAN_WEIGHTED': Path('../models/xgboost_fine_no_nan_weighted.pkl'),
    'NEURAL_NETWORK': Path('../models/neural_network.h5'),
    'NEURAL_NETWORK_NO_NAN': Path('../models/neural_network_no_nan.h5'),
    'AUTOML': Path('../models/automl.pkl'),
}

ENCODER = {
    'EYECOLOR': Path('../encoders/eyecolor_encoder.pkl'),
}

EVALUATION = {
    'LINEAR': Path('../evaluation/linear'),
    'LINEAR_NO_NAN': Path('../evaluation/linear_no_nan'),
    'LINEAR_NO_NAN_WEIGHTED': Path('../evaluation/linear_no_nan_weighted'),
    'XGBOOST': Path('../evaluation/xgboost'),
    'XGBOOST_NO_NAN': Path('../evaluation/xgboost_no_nan'),
    'XGBOOST_NO_NAN_WEIGHTED': Path('../evaluation/xgboost_no_nan_weighted'),
    'XGBOOST_FINE': Path('../evaluation/xgboost_fine'),
    'XGBOOST_FINE_NO_NAN': Path('../evaluation/xgboost_fine_no_nan'),
    'XGBOOST_FINE_NO_NAN_WEIGHTED': Path('../evaluation/xgboost_fine_no_nan_weighted'),
    'AUTOML': Path('../evaluation/automl'),
    'METRICS': Path('../evaluation/metrics.json'),
}