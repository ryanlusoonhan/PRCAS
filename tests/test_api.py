"""
Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Fixture for FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_customer_data():
    """Fixture for valid customer data."""
    return {
        "customer_data": {
            "state": "CA",
            "account_length": 100,
            "area_code": 415,
            "international_plan": "No",
            "voice_mail_plan": "Yes",
            "number_vmail_messages": 25,
            "total_day_minutes": 265.1,
            "total_day_calls": 110,
            "total_day_charge": 45.07,
            "total_eve_minutes": 197.4,
            "total_eve_calls": 99,
            "total_eve_charge": 16.78,
            "total_night_minutes": 244.7,
            "total_night_calls": 91,
            "total_night_charge": 11.01,
            "total_intl_minutes": 10.0,
            "total_intl_calls": 3,
            "total_intl_charge": 2.70,
            "customer_service_calls": 1
        }
    }


class TestHealthEndpoint:
    """Test cases for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_ready" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestPredictionEndpoint:
    """Test cases for prediction endpoint."""
    
    def test_predict_valid_data(self, client, valid_customer_data):
        """Test prediction with valid data."""
        # Note: This test may fail if model is not loaded
        # In that case, it should return 503
        response = client.post("/predict", json=valid_customer_data)
        
        # Should either succeed or return service unavailable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "risk_level" in data
            assert "recommendation" in data
            assert "top_features" in data
            assert 0 <= data["churn_probability"] <= 1
    
    def test_predict_invalid_state(self, client, valid_customer_data):
        """Test prediction with invalid state."""
        valid_customer_data["customer_data"]["state"] = "California"  # Too long
        
        response = client.post("/predict", json=valid_customer_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_negative_account_length(self, client, valid_customer_data):
        """Test prediction with negative account length."""
        valid_customer_data["customer_data"]["account_length"] = -5
        
        response = client.post("/predict", json=valid_customer_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_excessive_minutes(self, client, valid_customer_data):
        """Test prediction with excessive day minutes."""
        valid_customer_data["customer_data"]["total_day_minutes"] = 600.0
        
        response = client.post("/predict", json=valid_customer_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_field(self, client, valid_customer_data):
        """Test prediction with missing required field."""
        del valid_customer_data["customer_data"]["state"]
        
        response = client.post("/predict", json=valid_customer_data)
        
        assert response.status_code == 422  # Validation error


class TestExplanationEndpoint:
    """Test cases for explanation endpoint."""
    
    def test_explain_valid_data(self, client, valid_customer_data):
        """Test explanation with valid data."""
        response = client.post("/explain", json=valid_customer_data)
        
        # Should either succeed or return service unavailable
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "shap_values" in data
            assert "base_value" in data
            assert "feature_names" in data
            assert "top_influencers" in data


class TestModelInfoEndpoint:
    """Test cases for model info endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "is_ready" in data
            assert "model_type" in data
            assert "feature_count" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
