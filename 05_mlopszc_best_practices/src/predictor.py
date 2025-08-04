"""
Prediction service for spine disease classification.

This module provides a prediction service that loads trained models and
makes predictions on new data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
import joblib
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpineDiseasePredictor:
    """Prediction service for spine disease classification."""

    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the fitted scaler (optional)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            "pelvic_incidence",
            "pelvic_tilt",
            "lumbar_lordosis_angle",
            "sacral_slope",
            "pelvic_radius",
            "degree_spondylolisthesis",
        ]
        self.target_mapping = {0: "Normal", 1: "Disk Hernia", 2: "Spondylolisthesis"}
        
        self._load_model()
        if scaler_path:
            self._load_scaler()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def _load_scaler(self) -> None:
        """Load the fitted scaler from disk."""
        try:
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Scaler loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading scaler: {str(e)}")

    def validate_input(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and preprocess input data.

        Args:
            data: Input data in various formats

        Returns:
            Preprocessed numpy array

        Raises:
            ValueError: If input data is invalid
        """
        logger.info("Validating input data...")
        
        if isinstance(data, dict):
            # Convert dict to array
            features = []
            for feature in self.feature_names:
                if feature not in data:
                    raise ValueError(f"Missing required feature: {feature}")
                features.append(data[feature])
            data_array = np.array(features).reshape(1, -1)
            
        elif isinstance(data, pd.DataFrame):
            # Check if DataFrame has required columns
            missing_cols = set(self.feature_names) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            data_array = data[self.feature_names].values
            
        elif isinstance(data, np.ndarray):
            data_array = data
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
                
        else:
            raise ValueError("Input data must be dict, DataFrame, or numpy array")
        
        # Check shape
        if data_array.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {data_array.shape[1]}")
        
        # Check for missing values
        if np.isnan(data_array).any():
            raise ValueError("Input data contains missing values")
        
        logger.info(f"Input validation passed. Shape: {data_array.shape}")
        return data_array

    def preprocess_input(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data (scaling if scaler is available).

        Args:
            data: Input data array

        Returns:
            Preprocessed data array
        """
        if self.scaler is not None:
            logger.info("Scaling input data...")
            data_scaled = self.scaler.transform(data)
            return data_scaled
        else:
            logger.info("No scaler available, using raw data")
            return data

    def predict(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions on input data.

        Args:
            data: Input data in various formats

        Returns:
            Dictionary containing predictions and probabilities
        """
        logger.info("Making predictions...")
        
        # Validate and preprocess input
        data_array = self.validate_input(data)
        data_processed = self.preprocess_input(data_array)
        
        # Make predictions
        predictions = self.model.predict(data_processed)
        probabilities = self.model.predict_proba(data_processed)
        
        # Convert predictions to class names
        class_predictions = [self.target_mapping[pred] for pred in predictions]
        
        # Format results
        results = {
            "predictions": class_predictions,
            "probabilities": probabilities.tolist(),
            "prediction_codes": predictions.tolist(),
        }
        
        # Add confidence scores
        max_probs = np.max(probabilities, axis=1)
        results["confidence_scores"] = max_probs.tolist()
        
        logger.info(f"Predictions completed: {class_predictions}")
        return results

    def predict_batch(self, data: Union[List[Dict], pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make predictions on a batch of data.

        Args:
            data: Batch of input data

        Returns:
            Dictionary containing batch predictions and probabilities
        """
        logger.info("Making batch predictions...")
        
        if isinstance(data, list):
            # Convert list of dicts to DataFrame
            data_df = pd.DataFrame(data)
            return self.predict(data_df)
        else:
            return self.predict(data)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_path": self.model_path,
            "scaler_path": self.scaler_path,
            "feature_names": self.feature_names,
            "target_mapping": self.target_mapping,
            "model_type": type(self.model).__name__,
        }
        
        # Add model-specific information
        if hasattr(self.model, "feature_importances_"):
            info["has_feature_importance"] = True
        else:
            info["has_feature_importance"] = False
            
        return info

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the predictor.

        Returns:
            Dictionary containing health check results
        """
        health_status = {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "feature_count": len(self.feature_names),
            "target_classes": list(self.target_mapping.values()),
        }
        
        # Test prediction with sample data
        try:
            sample_data = {
                "pelvic_incidence": 63.0,
                "pelvic_tilt": 22.0,
                "lumbar_lordosis_angle": 39.0,
                "sacral_slope": 40.0,
                "pelvic_radius": 98.0,
                "degree_spondylolisthesis": 0.0,
            }
            test_prediction = self.predict(sample_data)
            health_status["prediction_test"] = "passed"
            health_status["sample_prediction"] = test_prediction
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["prediction_test"] = "failed"
            health_status["error"] = str(e)
        
        return health_status

    def save_prediction_log(self, predictions: Dict[str, Any], output_path: str) -> None:
        """
        Save prediction results to a log file.

        Args:
            predictions: Prediction results
            output_path: Path to save the log
        """
        logger.info(f"Saving prediction log to {output_path}")
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp and model info
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_info": self.get_model_info(),
            "predictions": predictions,
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        logger.info("Prediction log saved successfully")


class PredictionService:
    """High-level prediction service with additional features."""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize the prediction service.

        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.predictor = SpineDiseasePredictor(model_path, scaler_path)
        self.prediction_logs = []
    
    def predict_single(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a single prediction.

        Args:
            data: Single prediction input

        Returns:
            Prediction results
        """
        return self.predictor.predict(data)
    
    def predict_batch(self, data: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Make batch predictions.

        Args:
            data: List of prediction inputs

        Returns:
            Batch prediction results
        """
        return self.predictor.predict_batch(data)
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information.

        Returns:
            Service information
        """
        return {
            "service_name": "Spine Disease Classification Service",
            "version": "1.0.0",
            "model_info": self.predictor.get_model_info(),
            "health_status": self.predictor.health_check(),
        } 