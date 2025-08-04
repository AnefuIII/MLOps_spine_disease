"""
Model training utilities for spine disease classification.

This module handles model training, evaluation, and persistence for the
spine disease classification task.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation for spine disease classification."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.feature_names = None

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest classifier...")
        
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        
        rf_model = RandomForestClassifier(**default_params)
        rf_model.fit(X_train, y_train)
        
        self.models["random_forest"] = rf_model
        logger.info("Random Forest training completed")
        
        return rf_model

    def train_logistic_regression(
        self, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> LogisticRegression:
        """
        Train a Logistic Regression classifier.

        Args:
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional parameters for LogisticRegression

        Returns:
            Trained Logistic Regression model
        """
        logger.info("Training Logistic Regression classifier...")
        
        default_params = {
            "random_state": self.random_state,
            "max_iter": 1000,
            "multi_class": "multinomial",
        }
        default_params.update(kwargs)
        
        lr_model = LogisticRegression(**default_params)
        lr_model.fit(X_train, y_train)
        
        self.models["logistic_regression"] = lr_model
        logger.info("Logistic Regression training completed")
        
        return lr_model

    def hyperparameter_tuning(
        self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = "random_forest"
    ) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to tune ("random_forest" or "logistic_regression")

        Returns:
            Best model with optimized hyperparameters
        """
        logger.info(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_type == "logistic_regression":
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            }
            base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.models[f"{model_type}_tuned"] = best_model
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return best_model

    def evaluate_model(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for logging

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        precision = classification_rep["weighted avg"]["precision"]
        recall = classification_rep["weighted avg"]["recall"]
        f1_score = classification_rep["weighted avg"]["f1-score"]
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "classification_report": classification_rep,
            "confusion_matrix": conf_matrix.tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_pred_proba.tolist(),
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        logger.info(f"{model_name} - Precision: {precision:.4f}")
        logger.info(f"{model_name} - Recall: {recall:.4f}")
        logger.info(f"{model_name} - F1-Score: {f1_score:.4f}")
        
        return metrics

    def cross_validate_model(
        self, model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            cv: Number of cross-validation folds

        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        
        results = {
            "mean_cv_score": cv_scores.mean(),
            "std_cv_score": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results

    def select_best_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, str]:
        """
        Select the best performing model based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Tuple of (best_model, best_model_name)
        """
        logger.info("Selecting best model based on validation performance...")
        
        best_model = None
        best_model_name = None
        best_score = 0.0
        
        for model_name, model in self.models.items():
            score = model.score(X_val, y_val)
            logger.info(f"{model_name} validation score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name
        
        self.best_model = best_model
        self.best_score = best_score
        
        logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
        
        return best_model, best_model_name

    def save_model(self, model: Any, file_path: str) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Model to save
            file_path: Path where to save the model
        """
        logger.info(f"Saving model to {file_path}...")
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, file_path)
        logger.info("Model saved successfully")

    def load_model(self, file_path: str) -> Any:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {file_path}...")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        model = joblib.load(file_path)
        logger.info("Model loaded successfully")
        
        return model

    def get_feature_importance(self, model: Any, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance from a trained model.

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("Feature importance:")
        for feature, importance in sorted_importance.items():
            logger.info(f"  {feature}: {importance:.4f}")
        
        return sorted_importance

    def train_pipeline(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Complete model training pipeline.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary containing training results and best model
        """
        logger.info("Starting model training pipeline...")
        
        # Train multiple models
        rf_model = self.train_random_forest(X_train, y_train)
        lr_model = self.train_logistic_regression(X_train, y_train)
        
        # Hyperparameter tuning
        rf_tuned = self.hyperparameter_tuning(X_train, y_train, "random_forest")
        
        # Cross-validation
        rf_cv_results = self.cross_validate_model(rf_model, X_train, y_train)
        lr_cv_results = self.cross_validate_model(lr_model, X_train, y_train)
        rf_tuned_cv_results = self.cross_validate_model(rf_tuned, X_train, y_train)
        
        # Select best model
        best_model, best_model_name = self.select_best_model(X_val, y_val)
        
        # Evaluate best model
        best_model_metrics = self.evaluate_model(best_model, X_val, y_val, best_model_name)
        
        # Get feature importance if available
        feature_importance = self.get_feature_importance(best_model)
        
        results = {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_model_metrics": best_model_metrics,
            "feature_importance": feature_importance,
            "cross_validation_results": {
                "random_forest": rf_cv_results,
                "logistic_regression": lr_cv_results,
                "random_forest_tuned": rf_tuned_cv_results,
            },
            "all_models": self.models,
        }
        
        logger.info("Model training pipeline completed successfully")
        return results 