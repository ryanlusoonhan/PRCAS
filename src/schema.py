"""
Pandera DataFrame Schema Definitions for Data Validation

This module implements strict data validation schemas as specified in the Tech Spec.
All input data must pass validation before processing.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import yaml
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


class DataValidator:
    """
    Data validation class using Pandera schemas.
    Implements validation rules as specified in TSD section 1.2
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize validator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.validation_rules = self.config.get('validation', {})
        self.rejects_path = self.config.get('data', {}).get('rejects_path', 'rejects/')
        
        # Ensure rejects directory exists
        os.makedirs(self.rejects_path, exist_ok=True)
        
        # Initialize schema
        self.schema = self._build_schema()
        logger.info("DataValidator initialized successfully")
    
    def _build_schema(self) -> DataFrameSchema:
        """
        Build Pandera DataFrameSchema based on configuration.
        
        Returns:
            Configured DataFrameSchema
        """
        columns = {}
        
        # Account Length - must be positive integer
        if 'account_length' in self.validation_rules:
            rules = self.validation_rules['account_length']
            columns['Account length'] = Column(
                int,
                [
                    Check.greater_than(rules.get('min', 0)),
                    Check.less_than_or_equal_to(rules.get('max', 250)),
                ],
                nullable=False
            )
        
        # Total Day Minutes - non-negative float with range
        if 'total_day_minutes' in self.validation_rules:
            rules = self.validation_rules['total_day_minutes']
            columns['Total day minutes'] = Column(
                float,
                [
                    Check.greater_than_or_equal_to(rules.get('min', 0.0)),
                    Check.less_than_or_equal_to(rules.get('max', 500.0)),
                ],
                nullable=False
            )
        
        # Total Day Calls - non-negative integer
        if 'total_day_calls' in self.validation_rules:
            rules = self.validation_rules['total_day_calls']
            columns['Total day calls'] = Column(
                int,
                [
                    Check.greater_than_or_equal_to(rules.get('min', 0)),
                    Check.less_than_or_equal_to(rules.get('max', 200)),
                ],
                nullable=False
            )
        
        # Customer Service Calls - integer with range
        if 'customer_service_calls' in self.validation_rules:
            rules = self.validation_rules['customer_service_calls']
            columns['Customer service calls'] = Column(
                int,
                [
                    Check.greater_than_or_equal_to(rules.get('min', 0)),
                    Check.less_than_or_equal_to(rules.get('max', 20)),
                ],
                nullable=False
            )
        
        # Total Day Charge - non-negative float with range
        if 'total_day_charge' in self.validation_rules:
            rules = self.validation_rules['total_day_charge']
            columns['Total day charge'] = Column(
                float,
                [
                    Check.greater_than_or_equal_to(rules.get('min', 0.0)),
                    Check.less_than_or_equal_to(rules.get('max', 100.0)),
                ],
                nullable=False
            )
        
        # Churn target column (nullable for inference)
        columns['Churn'] = Column(
            bool,
            nullable=True
        )
        
        # Categorical columns - State
        columns['State'] = Column(
            str,
            [Check.str_length(min_value=2, max_value=2)],
            nullable=False
        )
        
        # International plan - binary categorical
        columns['International plan'] = Column(
            str,
            [Check.isin(['Yes', 'No', 'yes', 'no', 'YES', 'NO'])],
            nullable=False
        )
        
        # Voice mail plan - binary categorical
        columns['Voice mail plan'] = Column(
            str,
            [Check.isin(['Yes', 'No', 'yes', 'no', 'YES', 'NO'])],
            nullable=False
        )
        
        # Area code
        columns['Area code'] = Column(
            int,
            nullable=False
        )
        
        # Number vmail messages
        columns['Number vmail messages'] = Column(
            int,
            [Check.greater_than_or_equal_to(0)],
            nullable=False
        )
        
        # All charge columns must be non-negative
        charge_columns = [
            'Total eve charge',
            'Total night charge',
            'Total intl charge'
        ]
        for col in charge_columns:
            columns[col] = Column(
                float,
                [Check.greater_than_or_equal_to(0.0)],
                nullable=False
            )
        
        # All minutes columns must be non-negative
        minutes_columns = [
            'Total eve minutes',
            'Total night minutes',
            'Total intl minutes'
        ]
        for col in minutes_columns:
            columns[col] = Column(
                float,
                [Check.greater_than_or_equal_to(0.0)],
                nullable=False
            )
        
        # All calls columns must be non-negative integers
        calls_columns = [
            'Total eve calls',
            'Total night calls',
            'Total intl calls'
        ]
        for col in calls_columns:
            columns[col] = Column(
                int,
                [Check.greater_than_or_equal_to(0)],
                nullable=False
            )
        
        return DataFrameSchema(
            columns=columns,
            strict=False,  # Allow additional columns
            coerce=True    # Try to coerce types
        )
    
    def validate(
        self, 
        df: pd.DataFrame, 
        log_rejects: bool = True
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Validate DataFrame against schema.
        
        Args:
            df: Input DataFrame to validate
            log_rejects: Whether to log rejected records
            
        Returns:
            Tuple of (valid_dataframe, rejected_dataframe or None)
        """
        logger.info(f"Validating DataFrame with {len(df)} records")
        
        try:
            # Attempt validation
            validated_df = self.schema.validate(df, lazy=True)
            logger.info(f"Validation successful for all {len(validated_df)} records")
            return validated_df, None
            
        except pa.errors.SchemaErrors as err:
            logger.warning(f"Validation failed for {len(err.failure_cases)} records")
            
            # Get indices of failed records
            failed_indices = err.failure_cases['index'].unique()
            
            # Split into valid and rejected
            valid_df = df.drop(index=failed_indices).reset_index(drop=True)
            rejected_df = df.loc[failed_indices].copy()
            
            # Add error information to rejected records
            error_info = err.failure_cases.groupby('index').apply(
                lambda x: '; '.join([f"{row['column']}: {row['check']}" for _, row in x.iterrows()])
            ).to_dict()
            
            rejected_df['validation_errors'] = rejected_df.index.map(
                lambda idx: error_info.get(idx, 'Unknown error')
            )
            rejected_df['rejected_timestamp'] = datetime.now().isoformat()
            
            # Log rejected records if requested
            if log_rejects:
                self._log_rejects(rejected_df)
            
            logger.info(f"Returning {len(valid_df)} valid records, {len(rejected_df)} rejected")
            return valid_df, rejected_df
    
    def _log_rejects(self, rejected_df: pd.DataFrame) -> None:
        """
        Log rejected records to file.
        
        Args:
            rejected_df: DataFrame of rejected records
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reject_file = os.path.join(
            self.rejects_path, 
            f"rejected_records_{timestamp}.csv"
        )
        
        try:
            rejected_df.to_csv(reject_file, index=False)
            logger.info(f"Rejected records logged to {reject_file}")
        except Exception as e:
            logger.error(f"Failed to log rejected records: {e}")
    
    def validate_single_record(
        self, 
        record: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a single record (for API requests).
        
        Args:
            record: Dictionary containing single record data
            
        Returns:
            Tuple of (is_valid, error_message or None)
        """
        try:
            df = pd.DataFrame([record])
            self.schema.validate(df)
            return True, None
        except pa.errors.SchemaError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"


# Singleton instance for application-wide use
_validator_instance: Optional[DataValidator] = None


def get_validator(config_path: str = "config.yaml") -> DataValidator:
    """
    Get or create singleton DataValidator instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DataValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DataValidator(config_path)
    return _validator_instance
