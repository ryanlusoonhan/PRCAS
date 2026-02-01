"""
Preprocessing Pipeline Module

This module implements a class-based preprocessing pipeline using sklearn.pipeline.
It encapsulates category encoding and scaling to prevent data leakage.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import yaml
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


class StateToRegionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert State to Region based on US Census Bureau regions.
    """
    
    def __init__(self, state_to_region_map: Optional[Dict[str, str]] = None):
        """
        Initialize transformer.
        
        Args:
            state_to_region_map: Mapping of state abbreviations to regions
        """
        self.state_to_region_map = state_to_region_map or {}
        self.feature_names_out_: List[str] = []
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method - no fitting required for this transformer."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform State column to Region.
        
        Args:
            X: Input DataFrame with 'State' column
            
        Returns:
            DataFrame with 'Region' column added and 'State' removed
        """
        X = X.copy()
        
        if 'State' in X.columns:
            X['Region'] = X['State'].map(self.state_to_region_map)
            # Fill any unmapped states with 'Unknown'
            X['Region'] = X['Region'].fillna('Unknown')
            X = X.drop('State', axis=1)
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        return self.feature_names_out_


class BinaryEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to encode binary categorical columns (Yes/No -> 1/0).
    """
    
    def __init__(self, binary_columns: List[str]):
        """
        Initialize transformer.
        
        Args:
            binary_columns: List of binary categorical column names
        """
        self.binary_columns = binary_columns
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method - no fitting required."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform binary columns to numeric.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with binary columns encoded
        """
        X = X.copy()
        
        for col in self.binary_columns:
            if col in X.columns:
                X[col] = X[col].map({
                    'Yes': 1, 'No': 0,
                    'yes': 1, 'no': 0,
                    'YES': 1, 'NO': 0
                })
                # Fill any unmapped values with 0
                X[col] = X[col].fillna(0).astype(int)
        
        return X


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns.
    """
    
    def __init__(self, columns_to_drop: List[str]):
        """
        Initialize transformer.
        
        Args:
            columns_to_drop: List of column names to drop
        """
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit method - no fitting required."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with columns dropped
        """
        X = X.copy()
        cols_to_drop = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=cols_to_drop)


class PreprocessingPipeline:
    """
    Main preprocessing pipeline class.
    
    Encapsulates all preprocessing steps including:
    - State to Region conversion
    - Binary encoding (Yes/No -> 1/0)
    - Column dropping (high correlation features)
    - Numerical scaling
    - Categorical one-hot encoding
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.schema_config = self.config.get('schema', {})
        
        # Extract configuration
        self.categorical_columns = self.schema_config.get('categorical_columns', [])
        self.binary_columns = self.schema_config.get('binary_columns', [])
        self.numerical_columns = self.schema_config.get('numerical_columns', [])
        self.drop_columns = self.schema_config.get('drop_columns', [])
        self.state_to_region = self.config.get('state_to_region', {})
        
        # Pipeline will be built on first fit
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        
        logger.info("PreprocessingPipeline initialized")
    
    def _build_pipeline(self) -> Pipeline:
        """
        Build the preprocessing pipeline.
        
        The pipeline is designed to prevent data leakage by:
        1. First applying stateless transformations (State->Region, binary encoding)
        2. Then fitting scalers/encoders only on training data
        
        Returns:
            Configured sklearn Pipeline
        """
        # Step 1: Convert State to Region
        state_transformer = StateToRegionTransformer(self.state_to_region)
        
        # Step 2: Encode binary columns
        binary_encoder = BinaryEncoderTransformer(self.binary_columns)
        
        # Step 3: Drop high-correlation columns
        column_dropper = ColumnDropperTransformer(self.drop_columns)
        
        # Step 4: Build column transformer for scaling and encoding
        # Numerical pipeline: Impute -> Scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: Impute -> One-Hot Encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Compute actual numerical columns AFTER accounting for drop_columns
        # This is needed because drop_columns may include columns from numerical_columns
        actual_numerical_columns = [
            col for col in self.numerical_columns 
            if col not in self.drop_columns and col not in self.binary_columns
        ]
        
        logger.debug(f"Numerical columns for transformer: {actual_numerical_columns}")
        
        # Combine numerical and categorical pipelines
        # Note: Region is created by state_transformer, so we reference it after transformation
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, actual_numerical_columns),
            ('cat', categorical_pipeline, ['Region'])  # Region is the main categorical column after transformation
        ], remainder='drop')  # Drop other columns that aren't explicitly handled
        
        # Build full pipeline
        pipeline = Pipeline([
            ('state_to_region', state_transformer),
            ('binary_encoder', binary_encoder),
            ('column_dropper', column_dropper),
            ('preprocessor', preprocessor)
        ])
        
        return pipeline
    
    def fit(self, X: pd.DataFrame, y=None) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline.
        
        IMPORTANT: This should ONLY be called on training data to prevent data leakage.
        
        Args:
            X: Training DataFrame
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessing pipeline on {len(X)} records")
        
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X)
        
        # Extract feature names after transformation
        self._extract_feature_names(X)
        
        logger.info(f"Pipeline fitted. Output features: {len(self.feature_names)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed numpy array
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        logger.debug(f"Transforming {len(X)} records")
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Input DataFrame
            y: Target variable (optional)
            
        Returns:
            Transformed numpy array
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _extract_feature_names(self, X: pd.DataFrame) -> None:
        """
        Extract feature names after transformation.
        
        Args:
            X: Sample DataFrame to determine feature names
        """
        # Get preprocessor from pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        
        # Get numerical feature names (they stay the same)
        num_features = self.numerical_columns
        
        # Get categorical feature names from one-hot encoder
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(['Region']).tolist()
        
        # Binary columns that remain (not dropped)
        binary_remaining = [
            col for col in self.binary_columns 
            if col not in self.drop_columns
        ]
        
        # Combine all feature names
        self.feature_names = num_features + cat_features + binary_remaining
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.feature_names:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        return self.feature_names
    
    def save(self, filepath: str) -> None:
        """
        Save pipeline to disk.
        
        Args:
            filepath: Path to save pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Nothing to save.")
        
        # Save both pipeline and feature names
        artifact = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'config': {
                'numerical_columns': self.numerical_columns,
                'binary_columns': self.binary_columns,
                'drop_columns': self.drop_columns
            }
        }
        
        joblib.dump(artifact, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load(self, filepath: str) -> 'PreprocessingPipeline':
        """
        Load pipeline from disk.
        
        Args:
            filepath: Path to saved pipeline
            
        Returns:
            Self for method chaining
        """
        artifact = joblib.load(filepath)
        
        self.pipeline = artifact['pipeline']
        self.feature_names = artifact['feature_names']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return self
    
    def prepare_inference_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare single record for inference.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            DataFrame ready for transformation
        """
        df = pd.DataFrame([data])
        return df


def create_preprocessing_pipeline(config_path: str = "config.yaml") -> PreprocessingPipeline:
    """
    Factory function to create preprocessing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured PreprocessingPipeline instance
    """
    return PreprocessingPipeline(config_path)
