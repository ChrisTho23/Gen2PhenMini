import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from config import DATA, ENCODER

def encode_NaN(df):
    df = df.copy()
    df.fillna(3, inplace=True)
    return df

def drop_insignificant(df, columns_to_drop):
    df = df.copy()
    df.drop(columns=columns_to_drop, inplace=True)
    return df

class LabelEncoderWrapper:
    def __init__(self):
        self.encoder = LabelEncoder()
    
    def fit_transform(self, labels):
        return self.encoder.fit_transform(labels)
    
    def inverse_transform(self, encoded_labels):
        return self.encoder.inverse_transform(encoded_labels)



if __name__ == "__main__":
    print("Preprocessing data...")

    columns_to_drop = ['rs12203592', 'rs6119471']

    df = pd.read_csv(DATA['ENCODED'])

    eyecolor_encoder = LabelEncoderWrapper()
    df['eye_color'] = eyecolor_encoder.fit_transform(df['eye_color'])
    joblib.dump(eyecolor_encoder, ENCODER['EYECOLOR'])

    df = drop_insignificant(df, columns_to_drop)
    df_encoded = encode_NaN(df)
    df_encoded_no_nan = df.dropna()

    df_encoded.to_csv(DATA['PREPROCESSED'], index=False)
    df_encoded_no_nan.to_csv(DATA['PREPROCESSED_NO_NAN'], index=False)

    print("Preprocessing complete!")