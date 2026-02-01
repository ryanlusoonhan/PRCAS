"""
Predictor Module - Singleton Pattern for Model Inference

This module implements a singleton-pattern class to load saved artifacts
and perform inference with exception handling.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import joblib
import yaml
import shap

from src.preprocessing import PreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Data class for prediction results."""
    churn_probability: float
    risk_level: str
    recommendation: str
    top_features: List[Dict[str, Any]]
    prediction: int


@dataclass
class ExplanationResult:
    """Data class for SHAP explanation results."""
    shap_values: List[float]
    base_value: float
    feature_names: List[str]
    feature_values: List[float]
    top_influencers: List[Dict[str, Any]]


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


class ChurnPredictor:
    """
    Singleton-pattern predictor class for churn prediction.
    
    Loads saved model and preprocessor artifacts and provides
    inference capabilities with SHAP explanations.
    """
    
    _instance: Optional['ChurnPredictor'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ChurnPredictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize predictor (only runs once due to singleton).
        
        Args:
            config_path: Path to configuration file
        """
        if ChurnPredictor._initialized:
            return
        
        self.config = load_config(config_path)
        self.model_config = self.config.get('models', {})
        self.prediction_config = self.config.get('prediction', {})
        
        # Model artifacts
        self.model: Optional[Any] = None
        self.preprocessor: Optional[PreprocessingPipeline] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        
        # Risk level thresholds
        self.risk_levels = self.prediction_config.get('risk_levels', {})
        self.default_threshold = self.prediction_config.get('default_threshold', 0.5)
        
        # Load artifacts
        self._load_artifacts()
        
        ChurnPredictor._initialized = True
        logger.info("ChurnPredictor initialized successfully")
    
    def _load_artifacts(self) -> None:
        """
        Load model and preprocessor artifacts from disk.
        """
        model_dir = self.model_config.get('model_dir', 'models/')
        
        try:
            # Load model
            model_path = os.path.join(model_dir, self.model_config.get('model_filename', 'xgboost_churn_model.pkl'))
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}")
            
            # Load preprocessor
            preprocessor_path = os.path.join(model_dir, self.model_config.get('transformer_filename', 'preprocessing_transformer.pkl'))
            if os.path.exists(preprocessor_path):
                self.preprocessor = PreprocessingPipeline()
                self.preprocessor.load(preprocessor_path)
                self.feature_names = self.preprocessor.get_feature_names()
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            else:
                logger.warning(f"Preprocessor file not found at {preprocessor_path}")
            
            # Initialize SHAP explainer
            if self.model is not None:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("SHAP TreeExplainer initialized")
                
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise RuntimeError(f"Failed to initialize predictor: {e}")
    
    def is_ready(self) -> bool:
        """
        Check if predictor is ready for inference.
        
        Returns:
            True if model and preprocessor are loaded
        """
        return self.model is not None and self.preprocessor is not None
    
    def _get_risk_level(self, probability: float) -> Tuple[str, str]:
        """
        Determine risk level and recommendation based on probability.
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Tuple of (risk_level, recommendation)
        """
        low_config = self.risk_levels.get('low', {'max': 0.3, 'label': 'Low', 'recommendation': 'Standard Monitoring'})
        medium_config = self.risk_levels.get('medium', {'min': 0.3, 'max': 0.7, 'label': 'Medium', 'recommendation': 'Preventive Outreach'})
        high_config = self.risk_levels.get('high', {'min': 0.7, 'label': 'High', 'recommendation': 'High Priority Retention Call'})
        
        if probability < low_config.get('max', 0.3):
            return low_config.get('label', 'Low'), low_config.get('recommendation', 'Standard Monitoring')
        elif probability < medium_config.get('max', 0.7):
            return medium_config.get('label', 'Medium'), medium_config.get('recommendation', 'Preventive Outreach')
        else:
            return high_config.get('label', 'High'), high_config.get('recommendation', 'High Priority Retention Call')
    
    def _get_top_features(
        self, 
        shap_values: np.ndarray, 
        feature_values: np.ndarray,
        n_top: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get top contributing features based on SHAP values.
        
        Args:
            shap_values: SHAP values array
            feature_values: Feature values array
            n_top: Number of top features to return
            
        Returns:
            List of top feature dictionaries
        """
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-n_top:][::-1]
        
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            impact = float(abs_shap[idx])
            value = float(feature_values[idx]) if idx < len(feature_values) else 0.0
            
            top_features.append({
                'feature': feature_name,
                'impact': round(impact, 4),
                'value': round(value, 4),
                'direction': 'increases' if shap_values[idx] > 0 else 'decreases'
            })
        
        return top_features
    
    def predict(self, data: Dict[str, Any]) -> PredictionResult:
        """
        Make churn prediction for a single record.
        
        Args:
            data: Dictionary containing customer features
            
        Returns:
            PredictionResult with probability and risk assessment
            
        Raises:
            RuntimeError: If predictor is not ready
            ValueError: If input data is invalid
        """
        if not self.is_ready():
            raise RuntimeError("Predictor not initialized. Model or preprocessor not loaded.")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Preprocess
            X = self.preprocessor.transform(df)
            
            # Predict
            probability = float(self.model.predict_proba(X)[0, 1])
            prediction = 1 if probability >= self.default_threshold else 0
            
            # Get risk level
            risk_level, recommendation = self._get_risk_level(probability)
            
            # Get SHAP values for top features
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use positive class
            
            top_features = self._get_top_features(shap_values[0], X[0])
            
            return PredictionResult(
                churn_probability=round(probability, 4),
                risk_level=risk_level,
                recommendation=recommendation,
                top_features=top_features,
                prediction=prediction
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Failed to make prediction: {e}")
    
    def predict_batch(self, data: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Make predictions for multiple records.
        
        Args:
            data: List of dictionaries containing customer features
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        for record in data:
            try:
                result = self.predict(record)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for record: {e}")
                # Return a default/error result
                results.append(PredictionResult(
                    churn_probability=0.0,
                    risk_level="Error",
                    recommendation=f"Prediction failed: {str(e)}",
                    top_features=[],
                    prediction=-1
                ))
        return results
    
    def explain(self, data: Dict[str, Any]) -> ExplanationResult:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            data: Dictionary containing customer features
            
        Returns:
            ExplanationResult with SHAP values and feature contributions
            
        Raises:
            RuntimeError: If predictor is not ready
            ValueError: If input data is invalid
        """
        if not self.is_ready():
            raise RuntimeError("Predictor not initialized. Model or preprocessor not loaded.")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])
            
            # Preprocess
            X = self.preprocessor.transform(df)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Get base value (expected value)
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
            
            # Get top influencers
            top_influencers = self._get_top_features(shap_values[0], X[0], n_top=5)
            
            return ExplanationResult(
                shap_values=shap_values[0].tolist(),
                base_value=base_value,
                feature_names=self.feature_names,
                feature_values=X[0].tolist(),
                top_influencers=top_influencers
            )
            
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise ValueError(f"Failed to generate explanation: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'is_ready': self.is_ready(),
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_names),
            'threshold': self.default_threshold,
            'risk_levels': self.risk_levels
        }
        
        if self.model and hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        return info
    
    def reload(self) -> None:
        """
        Reload model artifacts from disk.
        Useful for updating model without restarting service.
        """
        logger.info("Reloading model artifacts...")
        self._load_artifacts()
        logger.info("Model artifacts reloaded")


# Convenience function to get predictor instance
def get_predictor(config_path: str = "config.yaml") -> ChurnPredictor:
    """
    Get or create singleton ChurnPredictor instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ChurnPredictor instance
    """
    return ChurnPredictor(config_path)
