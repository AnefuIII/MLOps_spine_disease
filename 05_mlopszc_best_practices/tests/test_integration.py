"""
Integration tests for the complete MLOps pipeline.

This module contains end-to-end integration tests that verify the complete
workflow from data processing to model training to prediction.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import joblib
from pathlib import Path
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.predictor import SpineDiseasePredictor


class TestCompletePipeline(unittest.TestCase):
    """Integration tests for the complete MLOps pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample dataset
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            "pelvic_incidence": np.random.uniform(30, 80, n_samples),
            "pelvic_tilt": np.random.uniform(5, 30, n_samples),
            "lumbar_lordosis_angle": np.random.uniform(20, 60, n_samples),
            "sacral_slope": np.random.uniform(20, 60, n_samples),
            "pelvic_radius": np.random.uniform(90, 130, n_samples),
            "degree_spondylolisthesis": np.random.uniform(-5, 5, n_samples),
            "class": np.random.choice(["Normal", "Disk Hernia", "Spondylolisthesis"], n_samples)
        })
        
        # Save sample data
        self.data_file = os.path.join(self.temp_dir, "sample_data.csv")
        self.sample_data.to_csv(self.data_file, index=False)
        
        # Initialize components
        self.data_processor = DataProcessor(random_state=42)
        self.model_trainer = ModelTrainer(random_state=42)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_pipeline_end_to_end(self):
        """Test the complete pipeline from data to prediction."""
        # Step 1: Data Processing
        processed_data = self.data_processor.process_pipeline(self.data_file)
        
        # Verify data processing results
        self.assertIn("X_train", processed_data)
        self.assertIn("y_train", processed_data)
        self.assertIn("X_val", processed_data)
        self.assertIn("y_val", processed_data)
        self.assertIn("X_test", processed_data)
        self.assertIn("y_test", processed_data)
        
        # Step 2: Model Training
        training_results = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Verify training results
        self.assertIsNotNone(training_results["best_model"])
        self.assertIsNotNone(training_results["best_model_name"])
        self.assertIn("best_model_metrics", training_results)
        
        # Step 3: Model Persistence
        model_file = os.path.join(self.temp_dir, "best_model.pkl")
        scaler_file = os.path.join(self.temp_dir, "scaler.pkl")
        
        self.model_trainer.save_model(training_results["best_model"], model_file)
        joblib.dump(processed_data["scaler"], scaler_file)
        
        # Verify files exist
        self.assertTrue(os.path.exists(model_file))
        self.assertTrue(os.path.exists(scaler_file))
        
        # Step 4: Model Loading and Prediction
        predictor = SpineDiseasePredictor(model_file, scaler_file)
        
        # Test prediction with sample input
        sample_input = {
            "pelvic_incidence": 63.0,
            "pelvic_tilt": 22.0,
            "lumbar_lordosis_angle": 39.0,
            "sacral_slope": 40.0,
            "pelvic_radius": 98.0,
            "degree_spondylolisthesis": 0.0,
        }
        
        prediction_result = predictor.predict(sample_input)
        
        # Verify prediction results
        self.assertIn("predictions", prediction_result)
        self.assertIn("probabilities", prediction_result)
        self.assertIn("confidence_scores", prediction_result)
        
        # Verify prediction format
        self.assertIsInstance(prediction_result["predictions"], list)
        self.assertIsInstance(prediction_result["probabilities"], list)
        self.assertIsInstance(prediction_result["confidence_scores"], list)
        
        # Verify prediction is valid
        valid_classes = ["Normal", "Disk Hernia", "Spondylolisthesis"]
        self.assertIn(prediction_result["predictions"][0], valid_classes)

    def test_pipeline_with_different_data_sizes(self):
        """Test pipeline with different dataset sizes."""
        # Test with small dataset
        small_data = self.sample_data.head(20)
        small_data_file = os.path.join(self.temp_dir, "small_data.csv")
        small_data.to_csv(small_data_file, index=False)
        
        # Process small dataset
        processed_small = self.data_processor.process_pipeline(small_data_file)
        
        # Verify small dataset processing
        self.assertGreater(len(processed_small["X_train"]), 0)
        self.assertGreater(len(processed_small["X_val"]), 0)
        self.assertGreater(len(processed_small["X_test"]), 0)
        
        # Train model on small dataset
        training_results = self.model_trainer.train_pipeline(
            processed_small["X_train"],
            processed_small["y_train"],
            processed_small["X_val"],
            processed_small["y_val"]
        )
        
        # Verify training completed
        self.assertIsNotNone(training_results["best_model"])

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data."""
        # Create invalid data (missing required column)
        invalid_data = self.sample_data.drop(columns=["pelvic_incidence"])
        invalid_data_file = os.path.join(self.temp_dir, "invalid_data.csv")
        invalid_data.to_csv(invalid_data_file, index=False)
        
        # Test that pipeline fails gracefully
        with self.assertRaises(ValueError):
            self.data_processor.process_pipeline(invalid_data_file)

    def test_model_performance_consistency(self):
        """Test that model performance is consistent across runs."""
        # Process data
        processed_data = self.data_processor.process_pipeline(self.data_file)
        
        # Train model multiple times with same random state
        results1 = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Create new trainer with same random state
        trainer2 = ModelTrainer(random_state=42)
        results2 = trainer2.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Verify consistent results
        self.assertEqual(
            results1["best_model_metrics"]["accuracy"],
            results2["best_model_metrics"]["accuracy"]
        )

    def test_feature_importance_consistency(self):
        """Test that feature importance is consistent."""
        # Process data
        processed_data = self.data_processor.process_pipeline(self.data_file)
        
        # Train model
        training_results = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Get feature importance
        importance1 = self.model_trainer.get_feature_importance(
            training_results["best_model"],
            processed_data["feature_names"]
        )
        
        # Verify feature importance
        self.assertIsInstance(importance1, dict)
        self.assertEqual(len(importance1), len(processed_data["feature_names"]))
        
        # Verify importance values are reasonable
        for feature, importance in importance1.items():
            self.assertGreaterEqual(importance, 0.0)
            self.assertLessEqual(importance, 1.0)

    def test_prediction_consistency(self):
        """Test that predictions are consistent for same input."""
        # Process data and train model
        processed_data = self.data_processor.process_pipeline(self.data_file)
        training_results = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Save model and scaler
        model_file = os.path.join(self.temp_dir, "model.pkl")
        scaler_file = os.path.join(self.temp_dir, "scaler.pkl")
        
        self.model_trainer.save_model(training_results["best_model"], model_file)
        joblib.dump(processed_data["scaler"], scaler_file)
        
        # Create predictor
        predictor = SpineDiseasePredictor(model_file, scaler_file)
        
        # Test same input multiple times
        sample_input = {
            "pelvic_incidence": 63.0,
            "pelvic_tilt": 22.0,
            "lumbar_lordosis_angle": 39.0,
            "sacral_slope": 40.0,
            "pelvic_radius": 98.0,
            "degree_spondylolisthesis": 0.0,
        }
        
        prediction1 = predictor.predict(sample_input)
        prediction2 = predictor.predict(sample_input)
        
        # Verify predictions are identical
        self.assertEqual(prediction1["predictions"], prediction2["predictions"])
        self.assertEqual(prediction1["prediction_codes"], prediction2["prediction_codes"])

    def test_batch_prediction_consistency(self):
        """Test batch prediction consistency."""
        # Process data and train model
        processed_data = self.data_processor.process_pipeline(self.data_file)
        training_results = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Save model and scaler
        model_file = os.path.join(self.temp_dir, "model.pkl")
        scaler_file = os.path.join(self.temp_dir, "scaler.pkl")
        
        self.model_trainer.save_model(training_results["best_model"], model_file)
        joblib.dump(processed_data["scaler"], scaler_file)
        
        # Create predictor
        predictor = SpineDiseasePredictor(model_file, scaler_file)
        
        # Test batch prediction
        batch_input = [
            {
                "pelvic_incidence": 63.0,
                "pelvic_tilt": 22.0,
                "lumbar_lordosis_angle": 39.0,
                "sacral_slope": 40.0,
                "pelvic_radius": 98.0,
                "degree_spondylolisthesis": 0.0,
            },
            {
                "pelvic_incidence": 45.0,
                "pelvic_tilt": 15.0,
                "lumbar_lordosis_angle": 30.0,
                "sacral_slope": 30.0,
                "pelvic_radius": 110.0,
                "degree_spondylolisthesis": -2.0,
            }
        ]
        
        batch_result = predictor.predict_batch(batch_input)
        
        # Verify batch prediction results
        self.assertIn("predictions", batch_result)
        self.assertIn("probabilities", batch_result)
        self.assertEqual(len(batch_result["predictions"]), 2)
        self.assertEqual(len(batch_result["probabilities"]), 2)

    def test_health_check_integration(self):
        """Test health check functionality in integration."""
        # Process data and train model
        processed_data = self.data_processor.process_pipeline(self.data_file)
        training_results = self.model_trainer.train_pipeline(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"]
        )
        
        # Save model and scaler
        model_file = os.path.join(self.temp_dir, "model.pkl")
        scaler_file = os.path.join(self.temp_dir, "scaler.pkl")
        
        self.model_trainer.save_model(training_results["best_model"], model_file)
        joblib.dump(processed_data["scaler"], scaler_file)
        
        # Create predictor
        predictor = SpineDiseasePredictor(model_file, scaler_file)
        
        # Test health check
        health_status = predictor.health_check()
        
        # Verify health check results
        self.assertEqual(health_status["status"], "healthy")
        self.assertTrue(health_status["model_loaded"])
        self.assertTrue(health_status["scaler_loaded"])
        self.assertEqual(health_status["feature_count"], 6)
        self.assertEqual(len(health_status["target_classes"]), 3)
        self.assertEqual(health_status["prediction_test"], "passed")


if __name__ == "__main__":
    unittest.main() 