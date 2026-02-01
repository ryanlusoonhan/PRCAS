"""
Unit tests for preprocessing pipeline
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    PreprocessingPipeline,
    StateToRegionTransformer,
    BinaryEncoderTransformer,
    ColumnDropperTransformer
)


@pytest.fixture
def sample_data():
    """Fixture for sample customer data."""
    return pd.DataFrame({
        'State': ['CA', 'NY', 'TX', 'FL', 'IL'],
        'Account length': [100, 50, 75, 120, 90],
        'Area code': [415, 212, 713, 305, 312],
        'International plan': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Voice mail plan': ['No', 'Yes', 'No', 'Yes', 'No'],
        'Number vmail messages': [0, 25, 0, 30, 0],
        'Total day minutes': [265.1, 150.0, 200.0, 300.0, 180.0],
        'Total day calls': [110, 80, 95, 120, 85],
        'Total day charge': [45.07, 25.50, 34.00, 51.00, 30.60],
        'Total eve minutes': [197.4, 200.0, 180.0, 220.0, 190.0],
        'Total eve calls': [99, 100, 90, 110, 95],
        'Total eve charge': [16.78, 17.00, 15.30, 18.70, 16.15],
        'Total night minutes': [244.7, 250.0, 230.0, 280.0, 240.0],
        'Total night calls': [91, 95, 85, 100, 90],
        'Total night charge': [11.01, 11.25, 10.35, 12.60, 10.80],
        'Total intl minutes': [10.0, 8.0, 12.0, 15.0, 9.0],
        'Total intl calls': [3, 2, 4, 5, 3],
        'Total intl charge': [2.70, 2.16, 3.24, 4.05, 2.43],
        'Customer service calls': [1, 0, 2, 3, 1]
    })


class TestStateToRegionTransformer:
    """Test cases for StateToRegionTransformer."""
    
    def test_transform(self, sample_data):
        """Test state to region transformation."""
        state_to_region = {
            'CA': 'West',
            'NY': 'Northeast',
            'TX': 'South',
            'FL': 'South',
            'IL': 'Midwest'
        }
        
        transformer = StateToRegionTransformer(state_to_region)
        result = transformer.transform(sample_data)
        
        assert 'Region' in result.columns
        assert 'State' not in result.columns
        assert result['Region'].tolist() == ['West', 'Northeast', 'South', 'South', 'Midwest']
    
    def test_unknown_state(self, sample_data):
        """Test handling of unknown state."""
        state_to_region = {'CA': 'West'}  # Only CA mapped
        
        transformer = StateToRegionTransformer(state_to_region)
        result = transformer.transform(sample_data)
        
        # Unknown states should be filled with 'Unknown'
        assert result['Region'].iloc[1] == 'Unknown'


class TestBinaryEncoderTransformer:
    """Test cases for BinaryEncoderTransformer."""
    
    def test_encode_binary_columns(self, sample_data):
        """Test binary column encoding."""
        transformer = BinaryEncoderTransformer([
            'International plan',
            'Voice mail plan'
        ])
        
        result = transformer.transform(sample_data)
        
        assert result['International plan'].tolist() == [1, 0, 1, 0, 1]
        assert result['Voice mail plan'].tolist() == [0, 1, 0, 1, 0]
    
    def test_case_insensitive(self, sample_data):
        """Test case-insensitive encoding."""
        sample_data['International plan'] = ['YES', 'no', 'Yes', 'NO', 'yes']
        
        transformer = BinaryEncoderTransformer(['International plan'])
        result = transformer.transform(sample_data)
        
        assert result['International plan'].tolist() == [1, 0, 1, 0, 1]


class TestColumnDropperTransformer:
    """Test cases for ColumnDropperTransformer."""
    
    def test_drop_columns(self, sample_data):
        """Test column dropping."""
        transformer = ColumnDropperTransformer(['Area code', 'State'])
        result = transformer.transform(sample_data)
        
        assert 'Area code' not in result.columns
        assert 'State' not in result.columns
        assert 'Account length' in result.columns  # Not dropped
    
    def test_nonexistent_columns(self, sample_data):
        """Test dropping non-existent columns."""
        transformer = ColumnDropperTransformer(['NonExistent'])
        result = transformer.transform(sample_data)
        
        # Should not raise error, just skip
        assert len(result.columns) == len(sample_data.columns)


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline."""
    
    def test_pipeline_fit_transform(self, sample_data):
        """Test full pipeline fit and transform."""
        pipeline = PreprocessingPipeline()
        
        result = pipeline.fit_transform(sample_data)
        
        # Should return numpy array
        assert isinstance(result, np.ndarray)
        # Should have more columns due to one-hot encoding
        assert result.shape[0] == len(sample_data)
        assert result.shape[1] > len(sample_data.columns)
    
    def test_feature_names_after_fit(self, sample_data):
        """Test feature name extraction after fitting."""
        pipeline = PreprocessingPipeline()
        pipeline.fit(sample_data)
        
        feature_names = pipeline.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_transform_without_fit_raises(self, sample_data):
        """Test that transform without fit raises error."""
        pipeline = PreprocessingPipeline()
        
        with pytest.raises(ValueError, match="Pipeline not fitted"):
            pipeline.transform(sample_data)
    
    def test_save_and_load(self, sample_data, tmp_path):
        """Test pipeline save and load functionality."""
        pipeline = PreprocessingPipeline()
        pipeline.fit(sample_data)
        
        # Save
        save_path = tmp_path / "pipeline.pkl"
        pipeline.save(str(save_path))
        
        # Load
        new_pipeline = PreprocessingPipeline()
        new_pipeline.load(str(save_path))
        
        # Should be able to transform
        result = new_pipeline.transform(sample_data)
        assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
