
import pip
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from xgboost import XGBClassifier

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from mlflow.models.signature import infer_signature



mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Spine-disease-exp')
mlflow.sklearn.autolog()

url = "/mnt/c/Users/anefu/Desktop/AI/mlopszc_proj_orc/spine_mage_pipeline/data/Dataset_spine.csv"

def read_dataframe(url):
    print(f'reading the data from {url}')
    df = pd.read_csv(url)
    

    return df



def preprocessing(df):
    print('preprocessing the data')
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    df['class_att'] = df['class_att'].map({'Abnormal': 0, 'Normal': 1}).astype(int)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    print('training the model')

    best_params = {
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.1086,
    'subsample': 0.7775,
    'colsample_bytree': 0.6674,
    'gamma': 3.2692
    }

    print('Starting MLflow run')
    with mlflow.start_run(run_name="xgb_script"):
        # Initialize model with **unpacked** parameters
        final_model = XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        final_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        # Manual logging to ensure all metrics are captured
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_auc": roc_auc_score(y_test, y_proba),
            "test_f1": f1_score(y_test, y_pred)
        })
        
        with open('models/xgbclassifier.pkl', 'wb') as f_out:
            pickle.dump(final_model, f_out)

        mlflow.log_artifact(local_path = 'models/xgbclassifier.pkl', artifact_path = 'local_model')
     
    print("Final model trained and logged successfully!")

def main(url):
    df = read_dataframe(url)
    X_train, X_test, y_train, y_test = preprocessing(df)
    trained_model = train_model(X_train, X_test, y_train, y_test)

if __name__== '__main__':
    main(url)





