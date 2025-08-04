"""
Unit tests for the ModelTrainer class.

This module contains comprehensive unit tests for model training,
evaluation, and persistence functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.model_trainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.trainer = ModelTrainer(random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 6)
        self.y_train = np.random.randint(0, 3, 100)
        self.X_val = np.random.randn(20, 6)
        self.y_val = np.random.randint(0, 3, 20)
        self.X_test = np.random.randn(20, 6)
        self.y_test = np.random.randint(0, 3, 20)

    def test_init(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=123)
        
        self.assertEqual(trainer.random_state, 123)
        self.assertEqual(trainer.models, {})
        self.assertIsNone(trainer.best_model)
        self.assertEqual(trainer.best_score, 0.0)

    def test_train_random_forest(self):
        """Test Random Forest training."""
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Check that model is trained
        self.assertIsNotNone(rf_model)
        self.assertIn("random_forest", self.trainer.models)
        
        # Check that model can make predictions
        predictions = rf_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_train_logistic_regression(self):
        """Test Logistic Regression training."""
        lr_model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        
        # Check that model is trained
        self.assertIsNotNone(lr_model)
        self.assertIn("logistic_regression", self.trainer.models)
        
        # Check that model can make predictions
        predictions = lr_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_hyperparameter_tuning_random_forest(self):
        """Test hyperparameter tuning for Random Forest."""
        best_model = self.trainer.hyperparameter_tuning(
            self.X_train, self.y_train, "random_forest"
        )
        
        # Check that tuned model is created
        self.assertIsNotNone(best_model)
        self.assertIn("random_forest_tuned", self.trainer.models)
        
        # Check that model can make predictions
        predictions = best_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_hyperparameter_tuning_logistic_regression(self):
        """Test hyperparameter tuning for Logistic Regression."""
        best_model = self.trainer.hyperparameter_tuning(
            self.X_train, self.y_train, "logistic_regression"
        )
        
        # Check that tuned model is created
        self.assertIsNotNone(best_model)
        self.assertIn("logistic_regression_tuned", self.trainer.models)
        
        # Check that model can make predictions
        predictions = best_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_hyperparameter_tuning_invalid_model(self):
        """Test hyperparameter tuning with invalid model type."""
        with self.assertRaises(ValueError) as context:
            self.trainer.hyperparameter_tuning(self.X_train, self.y_train, "invalid_model")
        
        self.assertIn("Unsupported model type", str(context.exception))

    def test_evaluate_model(self):
        """Test model evaluation."""
        # Train a model first
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Evaluate the model
        metrics = self.trainer.evaluate_model(rf_model, self.X_test, self.y_test, "test_model")
        
        # Check that all expected metrics are present
        expected_keys = ["accuracy", "precision", "recall", "f1_score", "classification_report", "confusion_matrix"]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check metric values
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertLessEqual(metrics["precision"], 1.0)

    def test_cross_validate_model(self):
        """Test cross-validation functionality."""
        # Train a model first
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Perform cross-validation
        cv_results = self.trainer.cross_validate_model(rf_model, self.X_train, self.y_train, cv=3)
        
        # Check that all expected keys are present
        expected_keys = ["mean_cv_score", "std_cv_score", "cv_scores"]
        for key in expected_keys:
            self.assertIn(key, cv_results)
        
        # Check that cv_scores has the right length
        self.assertEqual(len(cv_results["cv_scores"]), 3)
        
        # Check that mean and std are reasonable
        self.assertGreaterEqual(cv_results["mean_cv_score"], 0.0)
        self.assertLessEqual(cv_results["mean_cv_score"], 1.0)
        self.assertGreaterEqual(cv_results["std_cv_score"], 0.0)

    def test_select_best_model(self):
        """Test best model selection."""
        # Train multiple models
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        lr_model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        
        # Select best model
        best_model, best_model_name = self.trainer.select_best_model(self.X_val, self.y_val)
        
        # Check that a model is selected
        self.assertIsNotNone(best_model)
        self.assertIsNotNone(best_model_name)
        self.assertIn(best_model_name, self.trainer.models)
        
        # Check that best_score is updated
        self.assertGreater(self.trainer.best_score, 0.0)

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train a model
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            self.trainer.save_model(rf_model, temp_file)
            
            # Check that file exists
            self.assertTrue(os.path.exists(temp_file))
            
            # Load model
            loaded_model = self.trainer.load_model(temp_file)
            
            # Check that loaded model is the same type
            self.assertEqual(type(loaded_model), type(rf_model))
            
            # Check that predictions are the same
            original_predictions = rf_model.predict(self.X_test)
            loaded_predictions = loaded_model.predict(self.X_test)
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_load_model_file_not_found(self):
        """Test loading non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            self.trainer.load_model("non_existent_model.pkl")

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Train a Random Forest model (which has feature importance)
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.trainer.get_feature_importance(rf_model)
        
        # Check that importance is returned
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check that importance values sum to 1 (for Random Forest)
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=5)

    def test_get_feature_importance_with_names(self):
        """Test feature importance with custom feature names."""
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]
        
        # Train a Random Forest model
        rf_model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        # Get feature importance with custom names
        importance = self.trainer.get_feature_importance(rf_model, feature_names)
        
        # Check that feature names are used
        for feature_name in feature_names:
            self.assertIn(feature_name, importance)

    def test_get_feature_importance_no_importance(self):
        """Test feature importance for model without importance attribute."""
        # Train a Logistic Regression model (which doesn't have feature_importances_)
        lr_model = self.trainer.train_logistic_regression(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.trainer.get_feature_importance(lr_model)
        
        # Should return empty dict
        self.assertEqual(importance, {})

    def test_train_pipeline(self):
        """Test complete training pipeline."""
        results = self.trainer.train_pipeline(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Check that all expected keys are present
        expected_keys = [
            "best_model", "best_model_name", "best_model_metrics",
            "feature_importance", "cross_validation_results", "all_models"
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that best model is selected
        self.assertIsNotNone(results["best_model"])
        self.assertIsNotNone(results["best_model_name"])
        
        # Check that metrics are calculated
        self.assertIsInstance(results["best_model_metrics"], dict)
        
        # Check that multiple models are trained
        self.assertGreater(len(results["all_models"]), 1)

    def test_train_pipeline_with_empty_data(self):
        """Test training pipeline with empty data."""
        empty_X = np.array([]).reshape(0, 6)
        empty_y = np.array([])
        
        with self.assertRaises(ValueError):
            self.trainer.train_pipeline(empty_X, empty_y, self.X_val, self.y_val)


if __name__ == "__main__":
    unittest.main() 