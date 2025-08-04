"""
Unit tests for the DataProcessor class.

This module contains comprehensive unit tests for data loading,
validation, preprocessing, and feature scaling functionality.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor(random_state=42)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            "pelvic_incidence": [63.0, 39.0, 68.0],
            "pelvic_tilt": [22.0, 10.0, 18.0],
            "lumbar_lordosis_angle": [39.0, 25.0, 50.0],
            "sacral_slope": [40.0, 26.0, 50.0],
            "pelvic_radius": [98.0, 114.0, 108.0],
            "degree_spondylolisthesis": [0.0, -4.0, 0.0],
            "class": ["Normal", "Disk Hernia", "Spondylolisthesis"]
        })

    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor(random_state=123)
        
        self.assertEqual(processor.random_state, 123)
        self.assertIsNotNone(processor.scaler)
        self.assertEqual(len(processor.feature_columns), 6)
        self.assertEqual(processor.target_column, "class")

    def test_load_data_success(self):
        """Test successful data loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loaded_data = self.processor.load_data(temp_file)
            
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(len(loaded_data), 3)
            self.assertEqual(len(loaded_data.columns), 7)
        finally:
            os.unlink(temp_file)

    def test_load_data_file_not_found(self):
        """Test data loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_data("non_existent_file.csv")

    def test_load_data_empty_file(self):
        """Test data loading with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.processor.load_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_validate_data_success(self):
        """Test successful data validation."""
        result = self.processor.validate_data(self.sample_data)
        self.assertTrue(result)

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        invalid_data = self.sample_data.drop(columns=["pelvic_incidence"])
        
        with self.assertRaises(ValueError) as context:
            self.processor.validate_data(invalid_data)
        
        self.assertIn("Missing required columns", str(context.exception))

    def test_validate_data_missing_values(self):
        """Test data validation with missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, "pelvic_incidence"] = np.nan
        
        # Should not raise an exception, just log a warning
        result = self.processor.validate_data(data_with_missing)
        self.assertTrue(result)

    def test_validate_data_invalid_target_values(self):
        """Test data validation with invalid target values."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, "class"] = "Invalid"
        
        with self.assertRaises(ValueError) as context:
            self.processor.validate_data(invalid_data)
        
        self.assertIn("Invalid target values", str(context.exception))

    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed_data = self.processor.preprocess_data(self.sample_data)
        
        # Check that target is encoded
        self.assertIn(0, processed_data["class"].values)
        self.assertIn(1, processed_data["class"].values)
        self.assertIn(2, processed_data["class"].values)
        
        # Check that original data is not modified
        self.assertIn("Normal", self.sample_data["class"].values)

    def test_preprocess_data_with_missing_values(self):
        """Test data preprocessing with missing values."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, "pelvic_incidence"] = np.nan
        
        processed_data = self.processor.preprocess_data(data_with_missing)
        
        # Check that missing values are filled
        self.assertFalse(processed_data["pelvic_incidence"].isnull().any())

    def test_split_data(self):
        """Test data splitting functionality."""
        processed_data = self.processor.preprocess_data(self.sample_data)
        train, val, test = self.processor.split_data(processed_data, test_size=0.33, val_size=0.5)
        
        # Check that splits are non-empty
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        
        # Check that all data is used
        total_samples = len(train) + len(val) + len(test)
        self.assertEqual(total_samples, len(processed_data))

    def test_scale_features(self):
        """Test feature scaling functionality."""
        processed_data = self.processor.preprocess_data(self.sample_data)
        train, val, test = self.processor.split_data(processed_data)
        
        X_train_scaled, X_val_scaled, X_test_scaled = self.processor.scale_features(train, val, test)
        
        # Check that scaled features are numpy arrays
        self.assertIsInstance(X_train_scaled, np.ndarray)
        self.assertIsInstance(X_val_scaled, np.ndarray)
        self.assertIsInstance(X_test_scaled, np.ndarray)
        
        # Check shapes
        self.assertEqual(X_train_scaled.shape[1], len(self.processor.feature_columns))
        self.assertEqual(X_val_scaled.shape[1], len(self.processor.feature_columns))
        self.assertEqual(X_test_scaled.shape[1], len(self.processor.feature_columns))

    def test_get_targets(self):
        """Test target extraction functionality."""
        processed_data = self.processor.preprocess_data(self.sample_data)
        train, val, test = self.processor.split_data(processed_data)
        
        y_train, y_val, y_test = self.processor.get_targets(train, val, test)
        
        # Check that targets are numpy arrays
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_val, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        
        # Check that targets contain valid values
        valid_targets = [0, 1, 2]
        for target_array in [y_train, y_val, y_test]:
            for target in target_array:
                self.assertIn(target, valid_targets)

    def test_process_pipeline(self):
        """Test complete data processing pipeline."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            result = self.processor.process_pipeline(temp_file)
            
            # Check that all expected keys are present
            expected_keys = [
                "X_train", "X_val", "X_test",
                "y_train", "y_val", "y_test",
                "feature_names", "scaler", "target_mapping"
            ]
            
            for key in expected_keys:
                self.assertIn(key, result)
            
            # Check data types
            self.assertIsInstance(result["X_train"], np.ndarray)
            self.assertIsInstance(result["y_train"], np.ndarray)
            self.assertIsInstance(result["feature_names"], list)
            self.assertIsInstance(result["target_mapping"], dict)
            
        finally:
            os.unlink(temp_file)

    def test_process_pipeline_invalid_data(self):
        """Test pipeline with invalid data."""
        invalid_data = self.sample_data.drop(columns=["pelvic_incidence"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            invalid_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.processor.process_pipeline(temp_file)
        finally:
            os.unlink(temp_file)

    @patch('src.data_processor.pd.read_csv')
    def test_load_data_exception_handling(self, mock_read_csv):
        """Test exception handling in data loading."""
        mock_read_csv.side_effect = Exception("Test exception")
        
        with self.assertRaises(ValueError) as context:
            self.processor.load_data("test.csv")
        
        self.assertIn("Failed to load data", str(context.exception))


if __name__ == "__main__":
    unittest.main() 