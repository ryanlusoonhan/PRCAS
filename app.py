"""
FastAPI Application - Churn-Shield AI API

This module implements the REST API with:
- POST /predict: Churn probability prediction
- POST /explain: SHAP-based explanations
- GET /health: Health check and model info
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import yaml

from src.predictor import get_predictor, ChurnPredictor
from src.schema import get_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()
api_config = config.get('api', {})

# Initialize FastAPI app
app = FastAPI(
    title="Churn-Shield AI API",
    description="Predictive Retention & Customer Analytics System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (singleton)
predictor: Optional[ChurnPredictor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    try:
        predictor = get_predictor()
        logger.info("Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        # Don't raise - allow app to start even if model isn't ready
        predictor = None


# Pydantic Models for Request/Response Validation

class CustomerData(BaseModel):
    """Customer data model for prediction requests."""
    state: str = Field(..., description="US State abbreviation (e.g., 'CA', 'NY')", min_length=2, max_length=2)
    account_length: int = Field(..., description="Account length in months", ge=1, le=250)
    area_code: int = Field(..., description="Area code")
    international_plan: str = Field(..., description="International plan (Yes/No)")
    voice_mail_plan: str = Field(..., description="Voice mail plan (Yes/No)")
    number_vmail_messages: int = Field(..., description="Number of voicemail messages", ge=0)
    total_day_minutes: float = Field(..., description="Total day minutes", ge=0, le=500)
    total_day_calls: int = Field(..., description="Total day calls", ge=0, le=200)
    total_day_charge: float = Field(..., description="Total day charge", ge=0, le=100)
    total_eve_minutes: float = Field(..., description="Total evening minutes", ge=0)
    total_eve_calls: int = Field(..., description="Total evening calls", ge=0)
    total_eve_charge: float = Field(..., description="Total evening charge", ge=0)
    total_night_minutes: float = Field(..., description="Total night minutes", ge=0)
    total_night_calls: int = Field(..., description="Total night calls", ge=0)
    total_night_charge: float = Field(..., description="Total night charge", ge=0)
    total_intl_minutes: float = Field(..., description="Total international minutes", ge=0)
    total_intl_calls: int = Field(..., description="Total international calls", ge=0)
    total_intl_charge: float = Field(..., description="Total international charge", ge=0)
    customer_service_calls: int = Field(..., description="Number of customer service calls", ge=0, le=20)
    
    @validator('international_plan', 'voice_mail_plan')
    def validate_binary(cls, v):
        """Validate binary fields."""
        if v.lower() not in ['yes', 'no']:
            raise ValueError('Must be Yes or No')
        return v.title()
    
    @validator('state')
    def validate_state(cls, v):
        """Validate state is uppercase."""
        return v.upper()


class PredictRequest(BaseModel):
    """Prediction request model."""
    customer_data: CustomerData


class FeatureImpact(BaseModel):
    """Feature impact model for response."""
    feature: str
    impact: float
    value: float
    direction: str


class PredictResponse(BaseModel):
    """Prediction response model."""
    churn_probability: float
    risk_level: str
    recommendation: str
    top_features: List[FeatureImpact]
    prediction: int
    model_version: str = "1.0.0"
    timestamp: str


class ExplainRequest(BaseModel):
    """Explanation request model."""
    customer_data: CustomerData


class TopInfluencer(BaseModel):
    """Top influencer model for explanation."""
    feature: str
    impact: float
    value: float
    direction: str


class ExplainResponse(BaseModel):
    """Explanation response model."""
    shap_values: List[float]
    base_value: float
    feature_names: List[str]
    feature_values: List[float]
    top_influencers: List[TopInfluencer]
    model_version: str = "1.0.0"
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_ready: bool
    model_type: Optional[str]
    feature_count: int
    version: str = "1.0.0"
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# Helper functions

def convert_to_dataframe(customer_data: CustomerData) -> Dict[str, Any]:
    """
    Convert Pydantic model to dictionary for predictor.
    
    Args:
        customer_data: CustomerData Pydantic model
        
    Returns:
        Dictionary with proper column names
    """
    data = customer_data.dict()
    
    # Map to expected column names
    column_mapping = {
        'account_length': 'Account length',
        'area_code': 'Area code',
        'international_plan': 'International plan',
        'voice_mail_plan': 'Voice mail plan',
        'number_vmail_messages': 'Number vmail messages',
        'total_day_minutes': 'Total day minutes',
        'total_day_calls': 'Total day calls',
        'total_day_charge': 'Total day charge',
        'total_eve_minutes': 'Total eve minutes',
        'total_eve_calls': 'Total eve calls',
        'total_eve_charge': 'Total eve charge',
        'total_night_minutes': 'Total night minutes',
        'total_night_calls': 'Total night calls',
        'total_night_charge': 'Total night charge',
        'total_intl_minutes': 'Total intl minutes',
        'total_intl_calls': 'Total intl calls',
        'total_intl_charge': 'Total intl charge',
        'customer_service_calls': 'Customer service calls'
    }
    
    result = {'State': data['state']}
    for key, value in data.items():
        if key in column_mapping:
            result[column_mapping[key]] = value
    
    return result


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Churn-Shield AI API",
        "version": "1.0.0",
        "description": "Predictive Retention & Customer Analytics System",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model information.
    """
    try:
        if predictor is None:
            return HealthResponse(
                status="degraded",
                model_ready=False,
                model_type=None,
                feature_count=0,
                timestamp=datetime.now().isoformat()
            )
        
        model_info = predictor.get_model_info()
        
        return HealthResponse(
            status="healthy" if model_info['is_ready'] else "degraded",
            model_ready=model_info['is_ready'],
            model_type=model_info['model_type'],
            feature_count=model_info['feature_count'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Prediction"]
)
async def predict(request: PredictRequest):
    """
    Predict churn probability for a customer.
    
    Returns churn probability, risk level, and top contributing features.
    """
    try:
        # Check if predictor is ready
        if predictor is None or not predictor.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Convert request to predictor format
        data = convert_to_dataframe(request.customer_data)
        
        # Make prediction
        result = predictor.predict(data)
        
        # Convert to response format
        return PredictResponse(
            churn_probability=result.churn_probability,
            risk_level=result.risk_level,
            recommendation=result.recommendation,
            top_features=[
                FeatureImpact(
                    feature=f['feature'],
                    impact=f['impact'],
                    value=f['value'],
                    direction=f['direction']
                ) for f in result.top_features
            ],
            prediction=result.prediction,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Explanation"]
)
async def explain(request: ExplainRequest):
    """
    Generate SHAP explanation for a customer.
    
    Returns SHAP values and feature contributions for interpretability.
    """
    try:
        # Check if predictor is ready
        if predictor is None or not predictor.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        # Convert request to predictor format
        data = convert_to_dataframe(request.customer_data)
        
        # Generate explanation
        result = predictor.explain(data)
        
        # Convert to response format
        return ExplainResponse(
            shap_values=result.shap_values,
            base_value=result.base_value,
            feature_names=result.feature_names,
            feature_values=result.feature_values,
            top_influencers=[
                TopInfluencer(
                    feature=f['feature'],
                    impact=f['impact'],
                    value=f['value'],
                    direction=f['direction']
                ) for f in result.top_influencers
            ],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get detailed model information.
    
    Returns model parameters and configuration.
    """
    try:
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = predictor.get_model_info()
        return {
            **info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload model artifacts from disk.
    
    Useful for updating model without restarting service.
    """
    try:
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Predictor not initialized"
            )
        
        predictor.reload()
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Error handlers

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }


# Main entry point
if __name__ == "__main__":
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
