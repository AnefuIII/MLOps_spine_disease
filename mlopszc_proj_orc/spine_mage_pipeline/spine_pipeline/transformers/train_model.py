# import mlflow
# import mlflow.xgboost
# import xgboost as xgb
# from sklearn.metrics import accuracy_score
# import numpy as np

# @transformer
# def train_model(data: dict) -> dict:
#     print('Training the model')

#     # Convert lists back to numpy arrays
#     X_train = np.array(data['X_train'])
#     y_train = np.array(data['y_train'])
#     X_test = np.array(data['X_test'])
#     y_test = np.array(data['y_test'])

#     # Set MLflow experiment (optional)
#     mlflow.set_experiment("spine_classification")

#     with mlflow.start_run():
#         model = xgb.XGBClassifier(
#             n_estimators=100,
#             max_depth=4,
#             learning_rate=0.1,
#             use_label_encoder=False,
#             eval_metric='logloss',
#             random_state=42
#         )

#         model.fit(X_train, y_train)

#         # Predict and evaluate
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         print(f"Test Accuracy: {acc:.4f}")

#         # Log parameters and metrics
#         mlflow.log_param("n_estimators", 100)
#         mlflow.log_param("max_depth", 4)
#         mlflow.log_param("learning_rate", 0.1)
#         mlflow.log_metric("accuracy", acc)

#         # Log model
#         mlflow.xgboost.log_model(model, artifact_path="model")

#     return {
#         "accuracy": acc,
#         "model_path": "model"
#     }

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import numpy as np

@transformer
def train_model(data: dict) -> dict:
    print('Training the model')

    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])

    mlflow.set_experiment("spine_classification")

    with mlflow.start_run():
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")

        # Log with signature and example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("accuracy", acc)

        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example
        )

    return {
        "accuracy": acc,
        "model_path": "model"
    }
