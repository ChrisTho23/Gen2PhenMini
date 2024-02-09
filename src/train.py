from sklearn.model_selection import train_test_split
import pandas as pd

from config import DATA

def test_train_split(df):
    # Splitting the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Splitting the train set into train and validation sets
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Returning the train, validation, and test sets
    return train_df, val_df, test_df
    

if __main__ == "__name__":
    print("Training model...")

    df = pd.read_csv(DATA['PREPROCESSED'])

    train_df, val_df, test_df = test_train_split(df)