import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

from config import DATA, ENCODER

def genotype_encoder(df, reference_alleles):
    print("Encoding genotypes...")
    encoded_df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original

    # Iterate over each column that starts with 'rs'
    for rs_col in list(reference_alleles.keys()):
        # Apply the encoding rule for each cell in the column
        encoded_df[rs_col] = df[rs_col].apply(lambda genotype: 
            '0' if not pd.isna(genotype) and genotype == 2 * reference_alleles[rs_col] else
            '1' if not pd.isna(genotype) and reference_alleles[rs_col] in genotype else
            '2' if not pd.isna(genotype) else
            np.nan
        )

    encoded_df.drop(columns=['user_name', 'user_id'], inplace=True)

    encoded_df.to_csv(DATA['ENCODED'], index=False)

    return encoded_df

def encode_NaN(df):
    df = df.copy()
    df.fillna(3.0, inplace=True)
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

    # based on https://www.ncbi.nlm.nih.gov/snp/?term=rs12913832
    reference_alleles = {
        "rs12896399": "G", 
        "rs12913832": "A", 
        "rs1545397": "A",  
        "rs16891982": "C", 
        "rs1426654": "A",  
        "rs885479": "G",   
        "rs6119471": "C",  
        "rs12203592": "C"  
    }

    feature_to_drop = ['rs12203592', 'rs6119471']
    target_class_to_drop = ['Mixed']
    feature_cols = ['rs12896399', 'rs12913832', 'rs1545397', 'rs16891982', 'rs1426654', 'rs885479']
    target_col = 'eye_color'

    df_clean = pd.read_csv(DATA['CLEAN'])

    df = genotype_encoder(df_clean, reference_alleles)

    df = df[~df[target_col].isin(target_class_to_drop)]

    eyecolor_encoder = LabelEncoderWrapper()
    df[target_col] = eyecolor_encoder.fit_transform(df[target_col])
    joblib.dump(eyecolor_encoder, ENCODER['EYECOLOR'])

    df = drop_insignificant(df, feature_to_drop)
    df_encoded = encode_NaN(df)
    df_encoded_no_nan = df.dropna(subset=feature_cols)

    df_encoded.to_csv(DATA['PREPROCESSED'], index=False)
    df_encoded_no_nan.to_csv(DATA['PREPROCESSED_NO_NAN'], index=False)

    df_encoded = pd.get_dummies(df_encoded, columns=feature_cols, drop_first=True)
    df_encoded_no_nan = pd.get_dummies(df_encoded_no_nan, columns=feature_cols, drop_first=True)

    df_encoded.to_csv(DATA['FINAL'], index=False)
    df_encoded_no_nan.to_csv(DATA['FINAL_NO_NAN'], index=False)

    print("Preprocessing complete!")