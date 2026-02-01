# Churn-Shield AI Platform

**Predictive Retention & Customer Analytics System (PRCAS)**

A production-grade, modular, and containerized ML system for customer churn prediction and segmentation in the telecommunications industry.

## 🎯 Overview

Churn-Shield AI is an end-to-end predictive analytics platform that identifies at-risk telecommunications customers. The system provides:

- **Churn Prediction**: XGBoost-based binary classification with probability scoring
- **Customer Segmentation**: K-Means clustering for behavioral personas
- **Explainability**: SHAP-based feature importance and local interpretability
- **What-If Simulator**: Interactive scenario analysis for retention strategies

## 🏗️ Architecture

The system follows a **Modular Monolith** architecture, containerized using Docker:

```
[ Data Source (CSV/DVC) ] 
       │
       ▼
[ Training Pipeline ] ───▶ [ MLflow Tracking Server ]
       │                            │
       ▼                            ▼
[ Model Registry (.pkl) ] ◀─── [ Logic/Version Control ]
       │
       ▼
[ FastAPI Backend ] ◀─────▶ [ Streamlit Frontend Dashboard ]
       │
       ▼
[ Client / User ]
```

## 📁 Project Structure

```
PRCAS/
├── app.py                    # FastAPI application
├── dashboard.py              # Streamlit UI
├── config.yaml               # Centralized configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Main API Dockerfile
├── Dockerfile.dashboard      # Dashboard Dockerfile
├── docker-compose.yml        # Multi-container orchestration
├── src/
│   ├── __init__.py
│   ├── schema.py            # Pandera data validation schemas
│   ├── preprocessing.py     # Preprocessing pipeline
│   ├── train.py             # Training script with MLflow
│   └── predictor.py         # Singleton predictor with SHAP
├── models/                   # Saved model artifacts
├── data/                     # Data directory (DVC-tracked)
├── logs/                     # Application logs
└── rejects/                  # Rejected records from validation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd PRCAS
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare data**:
```bash
mkdir -p data/raw
# Place churn-bigml-80.csv and churn-bigml-20.csv in data/raw/
```

4. **Train the model**:
```bash
python -m src.train
```

5. **Run the API**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

6. **Run the Dashboard** (in a new terminal):
```bash
streamlit run dashboard.py
```

### Docker Deployment

Deploy the entire stack with Docker Compose:

```bash
docker-compose up -d
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5000

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Churn
```bash
POST /predict
Content-Type: application/json

{
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
```

### Explain Prediction (SHAP)
```bash
POST /explain
Content-Type: application/json

# Same request body as /predict
```

## 🔧 Configuration

All configuration is centralized in [`config.yaml`](config.yaml):

- **Data paths**: DVC-ready data locations
- **Model hyperparameters**: XGBoost settings
- **SMOTE configuration**: Resampling parameters
- **Validation rules**: Pandera schema constraints
- **Risk thresholds**: Probability cutoffs for risk levels

## 🧪 Key Features

### 1. Data Validation (Pandera)
Strict schema enforcement with automatic rejection logging:
- Account length > 0
- Usage minutes non-negative
- Customer service calls as integers
- Invalid records logged to `rejects/` folder

### 2. Preprocessing Pipeline (Scikit-Learn)
Class-based pipeline preventing data leakage:
- State → Region conversion (US Census Bureau)
- Binary encoding (Yes/No → 1/0)
- StandardScaler for numerical features
- OneHotEncoder for categorical features

### 3. SMOTE Integration (Imbalanced-Learn)
Data leakage prevention through pipeline integration:
- SMOTE applied only during `fit()` on training split
- Never applied during `predict()` or cross-validation

### 4. MLflow Tracking
Complete experiment tracking:
- Hyperparameters logging
- Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Model artifacts versioning
- Registered model management

### 5. SHAP Explainability
Local interpretability for every prediction:
- TreeExplainer for XGBoost
- Feature contribution breakdown
- Top risk drivers identification

### 6. What-If Simulator
Interactive scenario analysis in Streamlit:
- Adjust usage metrics via sliders
- Real-time prediction updates
- Churn reduction recommendations

## 📈 Model Performance

Based on the research implementation:

| Metric | Value |
|--------|-------|
| Accuracy | 95.88% |
| Precision | 89.04% |
| Recall | 82.28% |
| F1 Score | 85.53% |

## 🛡️ Production Standards

- **No Hardcoding**: All paths and constants from `config.yaml`
- **Data Leakage Prevention**: SMOTE and Scaling fitted only on training split
- **Logging**: Comprehensive logging with rotation
- **Type Hinting**: Full type annotations
- **Docstrings**: Every function documented
- **Error Handling**: Try-except blocks with descriptive error returns

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/
```

## 📝 License

This project is for portfolio exhibition purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code passes `flake8` linting
- All tests pass
- Type hints are included
- Docstrings are added

## 📧 Contact

For questions or support, please open an issue on the repository.

---

**Built with ❤️ using FastAPI, XGBoost, SHAP, and Streamlit**
