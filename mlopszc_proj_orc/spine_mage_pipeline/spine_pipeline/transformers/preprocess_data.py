import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@transformer
def preprocess_data(df):
    print('Preprocessing the data')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['class_att'] = df['class_att'].map({'Abnormal': 0, 'Normal': 1}).astype(int)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert numpy arrays to lists for JSON serialization
    return {
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist()
    }

