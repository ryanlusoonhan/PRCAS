"""
Training Module for Churn Prediction Model

This module implements a robust training pipeline that:
- Ingests data using DVC-ready paths
- Validates data via schema.py
- Implements SMOTE via imblearn.pipeline to avoid leakage
- Logs parameters and metrics to MLflow
- Saves model and transformer artifacts
"""

import logging
import os
import warnings
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.schema import DataValidator
from src.preprocessing import PreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


class ModelTrainer:
    """
    Model training orchestrator class.
    
    Handles the complete training workflow including data validation,
    preprocessing, SMOTE resampling, model training, and MLflow logging.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.model_config = self.config.get('models', {})
        self.mlflow_config = self.config.get('mlflow', {})
        self.xgboost_config = self.config.get('xgboost', {})
        self.smote_config = self.config.get('smote', {})
        self.training_config = self.config.get('training', {})
        
        # Initialize components
        self.validator = DataValidator(config_path)
        self.preprocessor = PreprocessingPipeline(config_path)
        
        # Model and metrics storage
        self.model: Optional[xgb.XGBClassifier] = None
        self.metrics: Dict[str, float] = {}
        self.cv_results: Dict[str, Any] = {}
        
        # Setup MLflow
        self._setup_mlflow()
        
        logger.info("ModelTrainer initialized successfully")
    
    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        try:
            tracking_uri = self.mlflow_config.get('tracking_uri', 'http://localhost:5000')
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment_name = self.mlflow_config.get('experiment_name', 'churn_prediction')
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow configured with experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}. Continuing without MLflow logging.")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing data from DVC-ready paths.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = self.data_config.get('raw_train_path', 'data/raw/churn-bigml-80.csv')
        test_path = self.data_config.get('raw_test_path', 'data/raw/churn-bigml-20.csv')
        
        logger.info(f"Loading data from {train_path} and {test_path}")
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Loaded train data: {train_df.shape}, test data: {test_df.shape}")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_data(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate training and test data.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of validated (train_df, test_df)
        """
        logger.info("Validating data...")
        
        # Validate training data
        train_valid, train_rejected = self.validator.validate(train_df)
        if train_rejected is not None:
            logger.warning(f"Rejected {len(train_rejected)} training records")
        
        # Validate test data
        test_valid, test_rejected = self.validator.validate(test_df)
        if test_rejected is not None:
            logger.warning(f"Rejected {len(test_rejected)} test records")
        
        return train_valid, test_valid
    
    def prepare_data(
        self, 
        train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with train/validation split.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Separate features and target
        target_col = self.config.get('schema', {}).get('target_column', 'Churn')
        
        X = train_df.drop(columns=[target_col])
        y = train_df[target_col].astype(int)
        
        # Train/validation split
        test_size = self.training_config.get('test_size', 0.2)
        random_state = self.training_config.get('random_state', 42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Validation={len(X_val)}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_val, y_train, y_val
    
    def build_training_pipeline(self) -> ImbPipeline:
        """
        Build the complete training pipeline with SMOTE.
        
        CRITICAL: SMOTE is applied within the pipeline to ensure it only
        operates on the training split during cross-validation, preventing
        data leakage.
        
        Returns:
            Imbalanced-learn Pipeline
        """
        # Get XGBoost parameters
        xgb_params = {
            'objective': self.xgboost_config.get('objective', 'binary:logistic'),
            'eval_metric': self.xgboost_config.get('eval_metric', 'aucpr'),
            'random_state': self.xgboost_config.get('random_state', 42),
            'n_estimators': self.xgboost_config.get('n_estimators', 200),
            'learning_rate': self.xgboost_config.get('learning_rate', 0.1),
            'max_depth': self.xgboost_config.get('max_depth', 5),
            'subsample': self.xgboost_config.get('subsample', 0.8),
            'colsample_bytree': self.xgboost_config.get('colsample_bytree', 0.8),
        }
        
        # Create XGBoost classifier (SMOTE handles imbalance, so no scale_pos_weight needed)
        classifier = xgb.XGBClassifier(**xgb_params)
        
        # Create SMOTE resampler
        smote = SMOTE(
            random_state=self.smote_config.get('random_state', 42),
            k_neighbors=self.smote_config.get('k_neighbors', 5)
        )
        
        # Build pipeline: Preprocessing -> SMOTE -> Classifier
        # Note: Preprocessor is fitted separately to save the artifact
        pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform stratified cross-validation.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of CV results
        """
        logger.info("Performing cross-validation...")
        
        # Build pipeline for CV (includes preprocessing)
        pipeline = self.build_training_pipeline()
        
        # Preprocess data first
        X_processed = self.preprocessor.fit_transform(X)
        
        # Setup CV
        n_folds = self.training_config.get('cv_folds', 5)
        random_state = self.training_config.get('random_state', 42)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Perform CV
        cv_accuracy = cross_val_score(pipeline, X_processed, y, cv=skf, scoring='accuracy')
        cv_precision = cross_val_score(pipeline, X_processed, y, cv=skf, scoring='precision')
        cv_recall = cross_val_score(pipeline, X_processed, y, cv=skf, scoring='recall')
        cv_f1 = cross_val_score(pipeline, X_processed, y, cv=skf, scoring='f1')
        cv_roc_auc = cross_val_score(pipeline, X_processed, y, cv=skf, scoring='roc_auc')
        
        results = {
            'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std()},
            'precision': {'mean': cv_precision.mean(), 'std': cv_precision.std()},
            'recall': {'mean': cv_recall.mean(), 'std': cv_recall.std()},
            'f1': {'mean': cv_f1.mean(), 'std': cv_f1.std()},
            'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std()}
        }
        
        logger.info(f"CV F1 Score: {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
        
        return results
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        y_train: pd.Series, 
        y_val: pd.Series
    ) -> xgb.XGBClassifier:
        """
        Train the model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training target
            y_val: Validation target
            
        Returns:
            Trained XGBoost classifier
        """
        logger.info("Training model...")
        
        # Fit preprocessor on training data only
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        
        # Build and fit training pipeline
        pipeline = self.build_training_pipeline()
        pipeline.fit(X_train_processed, y_train)
        
        # Extract the classifier from pipeline
        self.model = pipeline.named_steps['classifier']
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_processed)
        y_prob = self.model.predict_proba(X_val_processed)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_prob)
        }
        
        logger.info(f"Validation Metrics: {self.metrics}")
        
        return self.model
    
    def log_to_mlflow(self) -> None:
        """Log parameters, metrics, and artifacts to MLflow."""
        try:
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params({
                    'model_type': 'XGBoost',
                    'objective': self.xgboost_config.get('objective'),
                    'n_estimators': self.xgboost_config.get('n_estimators'),
                    'learning_rate': self.xgboost_config.get('learning_rate'),
                    'max_depth': self.xgboost_config.get('max_depth'),
                    'subsample': self.xgboost_config.get('subsample'),
                    'colsample_bytree': self.xgboost_config.get('colsample_bytree'),
                    'smote_k_neighbors': self.smote_config.get('k_neighbors')
                })
                
                # Log metrics
                mlflow.log_metrics(self.metrics)
                
                # Log CV results
                for metric_name, values in self.cv_results.items():
                    mlflow.log_metric(f"cv_{metric_name}_mean", values['mean'])
                    mlflow.log_metric(f"cv_{metric_name}_std", values['std'])
                
                # Log model
                mlflow.xgboost.log_model(self.model, "model")
                
                # Log artifacts
                model_dir = self.model_config.get('model_dir', 'models/')
                mlflow.log_artifacts(model_dir)
                
                logger.info(f"Logged to MLflow run: {run.info.run_id}")
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def save_artifacts(self) -> None:
        """Save model and preprocessor artifacts."""
        model_dir = self.model_config.get('model_dir', 'models/')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, self.model_config.get('model_filename', 'xgboost_churn_model.pkl'))
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, self.model_config.get('transformer_filename', 'preprocessing_transformer.pkl'))
        self.preprocessor.save(preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete training workflow.
        
        Returns:
            Dictionary of training results
        """
        logger.info("=" * 50)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 50)
        
        # 1. Load data
        train_df, test_df = self.load_data()
        
        # 2. Validate data
        train_df, test_df = self.validate_data(train_df, test_df)
        
        # 3. Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(train_df)
        
        # 4. Cross-validation
        self.cv_results = self.cross_validate(
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val])
        )
        
        # 5. Train model
        self.train(X_train, X_val, y_train, y_val)
        
        # 6. Log to MLflow
        self.log_to_mlflow()
        
        # 7. Save artifacts
        self.save_artifacts()
        
        logger.info("=" * 50)
        logger.info("Training Pipeline Complete")
        logger.info("=" * 50)
        
        return {
            'metrics': self.metrics,
            'cv_results': self.cv_results,
            'model': self.model,
            'preprocessor': self.preprocessor
        }


def main():
    """Main entry point for training."""
    trainer = ModelTrainer()
    results = trainer.run()
    
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Validation Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Validation Precision: {results['metrics']['precision']:.4f}")
    print(f"Validation Recall: {results['metrics']['recall']:.4f}")
    print(f"Validation F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"Validation ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
