"""
Unit tests for data validation schema
"""

import pytest
import pandas as pd
from src.schema import DataValidator


@pytest.fixture
def validator():
    """Fixture for DataValidator instance."""
    return DataValidator("config.yaml")


@pytest.fixture
def valid_record():
    """Fixture for a valid customer record."""
    return {
        'State': 'CA',
        'Account length': 100,
        'Area code': 415,
        'International plan': 'No',
        'Voice mail plan': 'Yes',
        'Number vmail messages': 25,
        'Total day minutes': 265.1,
        'Total day calls': 110,
        'Total day charge': 45.07,
        'Total eve minutes': 197.4,
        'Total eve calls': 99,
        'Total eve charge': 16.78,
        'Total night minutes': 244.7,
        'Total night calls': 91,
        'Total night charge': 11.01,
        'Total intl minutes': 10.0,
        'Total intl calls': 3,
        'Total intl charge': 2.70,
        'Customer service calls': 1,
        'Churn': False
    }


class TestDataValidation:
    """Test cases for data validation."""
    
    def test_valid_data_passes(self, validator, valid_record):
        """Test that valid data passes validation."""
        df = pd.DataFrame([valid_record])
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 1
        assert rejected_df is None
    
    def test_negative_account_length_fails(self, validator, valid_record):
        """Test that negative account length is rejected."""
        valid_record['Account length'] = -5
        df = pd.DataFrame([valid_record])
        
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 0
        assert len(rejected_df) == 1
    
    def test_excessive_day_minutes_fails(self, validator, valid_record):
        """Test that excessive day minutes are rejected."""
        valid_record['Total day minutes'] = 600.0  # Over max of 500
        df = pd.DataFrame([valid_record])
        
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 0
        assert len(rejected_df) == 1
    
    def test_invalid_state_format(self, validator, valid_record):
        """Test that invalid state format is rejected."""
        valid_record['State'] = 'California'  # Should be 2 chars
        df = pd.DataFrame([valid_record])
        
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 0
        assert len(rejected_df) == 1
    
    def test_invalid_binary_value(self, validator, valid_record):
        """Test that invalid binary values are rejected."""
        valid_record['International plan'] = 'Maybe'  # Should be Yes/No
        df = pd.DataFrame([valid_record])
        
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 0
        assert len(rejected_df) == 1
    
    def test_batch_validation(self, validator, valid_record):
        """Test batch validation with mixed valid/invalid records."""
        records = [
            valid_record.copy(),
            {**valid_record, 'Account length': -10},  # Invalid
            {**valid_record, 'Total day minutes': 600},  # Invalid
            valid_record.copy()
        ]
        
        df = pd.DataFrame(records)
        valid_df, rejected_df = validator.validate(df)
        
        assert len(valid_df) == 2
        assert len(rejected_df) == 2
    
    def test_single_record_validation(self, validator, valid_record):
        """Test single record validation."""
        is_valid, error = validator.validate_single_record(valid_record)
        
        assert is_valid is True
        assert error is None
    
    def test_single_record_validation_failure(self, validator, valid_record):
        """Test single record validation failure."""
        valid_record['Account length'] = -5
        is_valid, error = validator.validate_single_record(valid_record)
        
        assert is_valid is False
        assert error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
