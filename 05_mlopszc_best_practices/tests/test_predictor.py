"""
Unit tests for the SpineDiseasePredictor class.

This module contains comprehensive unit tests for prediction functionality,
input validation, and model loading.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.predictor import SpineDiseasePredictor, PredictionService


class TestSpineDiseasePredictor(unittest.TestCase):
    """Test cases for SpineDiseasePredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary model for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        self.temp_scaler_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        
        # Create a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_sample = np.random.randn(100, 6)
        y_sample = np.random.randint(0, 3, 100)
        self.model.fit(X_sample, y_sample)
        
        # Create a simple scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_sample)
        
        # Save model and scaler
        joblib.dump(self.model, self.temp_model_file.name)
        joblib.dump(self.scaler, self.temp_scaler_file.name)
        
        # Sample input data
        self.sample_input = {
            "pelvic_incidence": 63.0,
            "pelvic_tilt": 22.0,
            "lumbar_lordosis_angle": 39.0,
            "sacral_slope": 40.0,
            "pelvic_radius": 98.0,
            "degree_spondylolisthesis": 0.0,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)
        if os.path.exists(self.temp_scaler_file.name):
            os.unlink(self.temp_scaler_file.name)

    def test_init_with_model_only(self):
        """Test predictor initialization with model only."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        
        self.assertEqual(predictor.model_path, self.temp_model_file.name)
        self.assertIsNone(predictor.scaler_path)
        self.assertIsNotNone(predictor.model)
        self.assertIsNone(predictor.scaler)

    def test_init_with_model_and_scaler(self):
        """Test predictor initialization with model and scaler."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        
        self.assertEqual(predictor.model_path, self.temp_model_file.name)
        self.assertEqual(predictor.scaler_path, self.temp_scaler_file.name)
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.scaler)

    def test_init_model_file_not_found(self):
        """Test initialization with non-existent model file."""
        with self.assertRaises(FileNotFoundError):
            SpineDiseasePredictor("non_existent_model.pkl")

    def test_init_scaler_file_not_found(self):
        """Test initialization with non-existent scaler file."""
        with self.assertRaises(FileNotFoundError):
            SpineDiseasePredictor(self.temp_model_file.name, "non_existent_scaler.pkl")

    def test_validate_input_dict(self):
        """Test input validation with dictionary input."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        result = predictor.validate_input(self.sample_input)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 6))

    def test_validate_input_dataframe(self):
        """Test input validation with DataFrame input."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        df_input = pd.DataFrame([self.sample_input])
        result = predictor.validate_input(df_input)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 6))

    def test_validate_input_numpy_array(self):
        """Test input validation with numpy array input."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        array_input = np.array(list(self.sample_input.values())).reshape(1, -1)
        result = predictor.validate_input(array_input)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 6))

    def test_validate_input_missing_features(self):
        """Test input validation with missing features."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        invalid_input = {"pelvic_incidence": 63.0}  # Missing other features
        
        with self.assertRaises(ValueError) as context:
            predictor.validate_input(invalid_input)
        
        self.assertIn("Missing required feature", str(context.exception))

    def test_validate_input_wrong_shape(self):
        """Test input validation with wrong shape."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        invalid_input = np.array([1, 2, 3, 4, 5])  # Only 5 features instead of 6
        
        with self.assertRaises(ValueError) as context:
            predictor.validate_input(invalid_input)
        
        self.assertIn("Expected 6 features", str(context.exception))

    def test_validate_input_missing_values(self):
        """Test input validation with missing values."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        invalid_input = self.sample_input.copy()
        invalid_input["pelvic_incidence"] = np.nan
        
        with self.assertRaises(ValueError) as context:
            predictor.validate_input(invalid_input)
        
        self.assertIn("contains missing values", str(context.exception))

    def test_validate_input_invalid_type(self):
        """Test input validation with invalid data type."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        invalid_input = "not a valid input"
        
        with self.assertRaises(ValueError) as context:
            predictor.validate_input(invalid_input)
        
        self.assertIn("must be dict, DataFrame, or numpy array", str(context.exception))

    def test_preprocess_input_with_scaler(self):
        """Test input preprocessing with scaler."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        input_array = np.array(list(self.sample_input.values())).reshape(1, -1)
        
        result = predictor.preprocess_input(input_array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, input_array.shape)

    def test_preprocess_input_without_scaler(self):
        """Test input preprocessing without scaler."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name)
        input_array = np.array(list(self.sample_input.values())).reshape(1, -1)
        
        result = predictor.preprocess_input(input_array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, input_array.shape)
        np.testing.assert_array_equal(result, input_array)

    def test_predict_dict_input(self):
        """Test prediction with dictionary input."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        result = predictor.predict(self.sample_input)
        
        # Check that all expected keys are present
        expected_keys = ["predictions", "probabilities", "prediction_codes", "confidence_scores"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types
        self.assertIsInstance(result["predictions"], list)
        self.assertIsInstance(result["probabilities"], list)
        self.assertIsInstance(result["prediction_codes"], list)
        self.assertIsInstance(result["confidence_scores"], list)
        
        # Check that prediction is valid
        self.assertIn(result["predictions"][0], ["Normal", "Disk Hernia", "Spondylolisthesis"])

    def test_predict_dataframe_input(self):
        """Test prediction with DataFrame input."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        df_input = pd.DataFrame([self.sample_input])
        result = predictor.predict(df_input)
        
        self.assertIn("predictions", result)
        self.assertIn("probabilities", result)

    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        batch_input = [self.sample_input, self.sample_input]
        result = predictor.predict_batch(batch_input)
        
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 2)

    def test_get_model_info(self):
        """Test model information retrieval."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        info = predictor.get_model_info()
        
        expected_keys = ["model_path", "scaler_path", "feature_names", "target_mapping", "model_type"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info["model_path"], self.temp_model_file.name)
        self.assertEqual(info["scaler_path"], self.temp_scaler_file.name)
        self.assertEqual(len(info["feature_names"]), 6)
        self.assertEqual(len(info["target_mapping"]), 3)

    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        importance = predictor.get_feature_importance()
        
        # Random Forest should have feature importance
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), 6)
        
        # Check that importance values sum to 1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=5)

    def test_health_check(self):
        """Test health check functionality."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        health = predictor.health_check()
        
        expected_keys = ["status", "model_loaded", "scaler_loaded", "feature_count", "target_classes"]
        for key in expected_keys:
            self.assertIn(key, health)
        
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["model_loaded"])
        self.assertTrue(health["scaler_loaded"])
        self.assertEqual(health["feature_count"], 6)
        self.assertEqual(len(health["target_classes"]), 3)

    def test_save_prediction_log(self):
        """Test prediction log saving."""
        predictor = SpineDiseasePredictor(self.temp_model_file.name, self.temp_scaler_file.name)
        predictions = predictor.predict(self.sample_input)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            log_file = f.name
        
        try:
            predictor.save_prediction_log(predictions, log_file)
            
            # Check that file exists
            self.assertTrue(os.path.exists(log_file))
            
            # Check that file contains valid JSON
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            expected_keys = ["timestamp", "model_info", "predictions"]
            for key in expected_keys:
                self.assertIn(key, log_data)
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)


class TestPredictionService(unittest.TestCase):
    """Test cases for PredictionService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary model for testing
        self.temp_model_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        
        # Create a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_sample = np.random.randn(100, 6)
        y_sample = np.random.randint(0, 3, 100)
        self.model.fit(X_sample, y_sample)
        
        # Save model
        joblib.dump(self.model, self.temp_model_file.name)
        
        # Sample input data
        self.sample_input = {
            "pelvic_incidence": 63.0,
            "pelvic_tilt": 22.0,
            "lumbar_lordosis_angle": 39.0,
            "sacral_slope": 40.0,
            "pelvic_radius": 98.0,
            "degree_spondylolisthesis": 0.0,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_model_file.name):
            os.unlink(self.temp_model_file.name)

    def test_init(self):
        """Test PredictionService initialization."""
        service = PredictionService(self.temp_model_file.name)
        
        self.assertIsNotNone(service.predictor)
        self.assertEqual(service.prediction_logs, [])

    def test_predict_single(self):
        """Test single prediction."""
        service = PredictionService(self.temp_model_file.name)
        result = service.predict_single(self.sample_input)
        
        self.assertIn("predictions", result)
        self.assertIn("probabilities", result)

    def test_predict_batch(self):
        """Test batch prediction."""
        service = PredictionService(self.temp_model_file.name)
        batch_input = [self.sample_input, self.sample_input]
        result = service.predict_batch(batch_input)
        
        self.assertIn("predictions", result)
        self.assertEqual(len(result["predictions"]), 2)

    def test_get_service_info(self):
        """Test service information retrieval."""
        service = PredictionService(self.temp_model_file.name)
        info = service.get_service_info()
        
        expected_keys = ["service_name", "version", "model_info", "health_status"]
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info["service_name"], "Spine Disease Classification Service")
        self.assertEqual(info["version"], "1.0.0")


if __name__ == "__main__":
    unittest.main() 