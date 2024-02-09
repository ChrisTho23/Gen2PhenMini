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
}

MODEL = {
    'LINEAR': Path('../models/logistic_regression.pkl'),
    'LINEAR_NO_NAN': Path('../models/logistic_regression_no_nan.pkl'),
    'XGBOOST': Path('../models/xgboost.pkl'),
    'XGBOOST_FINE': Path('../evaluation/xgboost_fine.pkl'),
    'NEURAL_NETWORK': Path('../models/neural_network.h5'),
}

ENCODER = {
    'EYECOLOR': Path('../encoders/eyecolor_encoder.pkl'),
}

EVALUATION = {
    'LINEAR': Path('../evaluation/linear'),
    'LINEAR_NO_NAN': Path('../evaluation/linear_no_nan'),
    'XGBOOST': Path('../evaluation/xgboost'),
    'XGBOOST_FINE': Path('../evaluation/xgboost_fine'),
    'METRICS': Path('../evaluation/metrics.json'),
}