"""
Data processing utilities for spine disease classification.

This module handles data loading, preprocessing, and validation for the
vertebral column dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing for spine disease classification."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the DataProcessor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = [
            "pelvic_incidence",
            "pelvic_tilt",
            "lumbar_lordosis_angle",
            "sacral_slope",
            "pelvic_radius",
            "degree_spondylolisthesis",
        ]
        self.target_column = "class"

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the spine disease dataset.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or malformed
        """
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            
            if data.empty:
                raise ValueError("Dataset is empty")
            
            logger.info(f"Loaded {len(data)} rows and {len(data.columns)} columns")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the dataset structure and content.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        logger.info("Validating dataset...")
        
        # Check required columns
        missing_columns = set(self.feature_columns + [self.target_column]) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_counts = data[self.feature_columns + [self.target_column]].isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0]}")
        
        # Check data types
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Check target values
        valid_targets = ["Normal", "Disk Hernia", "Spondylolisthesis"]
        invalid_targets = set(data[self.target_column].unique()) - set(valid_targets)
        if invalid_targets:
            raise ValueError(f"Invalid target values: {invalid_targets}")
        
        logger.info("Data validation passed")
        return True

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for model training.

        Args:
            data: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Handle missing values
        for col in self.feature_columns:
            if processed_data[col].isnull().sum() > 0:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
        
        # Encode target variable
        target_mapping = {"Normal": 0, "Disk Hernia": 1, "Spondylolisthesis": 2}
        processed_data[self.target_column] = processed_data[self.target_column].map(target_mapping)
        
        logger.info("Data preprocessing completed")
        return processed_data

    def split_data(
        self, data: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: Preprocessed DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            data, test_size=test_size, random_state=self.random_state, stratify=data[self.target_column]
        )
        
        # Second split: train vs validation
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=self.random_state, stratify=train_val[self.target_column]
        )
        
        logger.info(f"Train set: {len(train)} samples")
        logger.info(f"Validation set: {len(val)} samples")
        logger.info(f"Test set: {len(test)} samples")
        
        return train, val, test

    def scale_features(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.

        Args:
            train: Training data
            val: Validation data
            test: Test data

        Returns:
            Tuple of scaled feature arrays
        """
        logger.info("Scaling features...")
        
        # Fit scaler on training data
        X_train = train[self.feature_columns].values
        X_val = val[self.feature_columns].values
        X_test = test[self.feature_columns].values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_targets(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract target variables from datasets.

        Args:
            train: Training data
            val: Validation data
            test: Test data

        Returns:
            Tuple of target arrays
        """
        y_train = train[self.target_column].values
        y_val = val[self.target_column].values
        y_test = test[self.target_column].values
        
        return y_train, y_val, y_test

    def process_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Complete data processing pipeline.

        Args:
            file_path: Path to the data file

        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data processing pipeline...")
        
        # Load and validate data
        data = self.load_data(file_path)
        self.validate_data(data)
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Split data
        train, val, test = self.split_data(processed_data)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(train, val, test)
        
        # Get targets
        y_train, y_val, y_test = self.get_targets(train, val, test)
        
        result = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": self.feature_columns,
            "scaler": self.scaler,
            "target_mapping": {"Normal": 0, "Disk Hernia": 1, "Spondylolisthesis": 2},
        }
        
        logger.info("Data processing pipeline completed successfully")
        return result 