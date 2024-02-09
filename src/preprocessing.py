import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import DATA

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

    df = drop_insignificant(df, columns_to_drop)
    df = encode_NaN(df)

    eyecolor_encoder = LabelEncoderWrapper()
    df['eye_color'] = eyecolor_encoder.fit_transform(df['eye_color'])

    df.to_csv(DATA['PREPROCESSED'], index=False)

    print("Preprocessing complete!")