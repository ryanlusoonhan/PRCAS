# Technical Specification: Churn-Shield AI Platform

**Project:** Customer Analytics & Prediction System  
**Lead Engineer:** [Your Name]  
**Version:** 1.0  
**Stack:** Python 3.9+, XGBoost 1.7+, FastAPI 0.95+, Pandera, SHAP, Docker

---

## 1. Data Engineering Specification

### 1.1 Input Feature Engineering (Transformation Mapping)
| Raw Feature | Type | Transformation | Logic / Mathematical Basis |
| :--- | :--- | :--- | :--- |
| `State` | Categorical | One-Hot Encoding | Map $N$ states to $N-1$ binary columns. |
| `Intl_Plan` | Binary | Label Encoding | Yes $\rightarrow$ 1, No $\rightarrow$ 0. |
| `Day_Mins` | Float | Standardization | $z = \frac{x - \mu}{\sigma}$ (Z-score normalization). |
| `CustServ_Calls` | Integer | Pass-through | Kept as discrete count; vital for tree-splits. |
| `Total_Charge` | Numerical | Feature Synthesis | Sum of `Day`, `Eve`, `Night`, and `Intl` charges. |

### 1.2 Data Validation (Pandera Schema)
```python
schema = pa.DataFrameSchema({
    "Account_Length": pa.Column(int, pa.Check.greater_than(0)),
    "Total_Day_Mins": pa.Column(float, pa.Check.in_range(0, 500)),
    "CustServ_Calls": pa.Column(int, pa.Check.in_range(0, 20)),
    "Churn": pa.Column(int, pa.Check.isin([0, 1]), nullable=True)
})
```

---

## 2. Unsupervised Learning Spec (Segmentation)

### 2.1 K-Means Clustering
*   **Initialization:** K-Means++ (to avoid local optima).
*   **Distance Metric:** Euclidean Distance.
*   **Optimization:** Elbow Method (WCSS) + Silhouette Coefficient.
*   **Feature Scaling:** Min-Max Scaling [0,1] is required prior to clustering to ensure usage "minutes" don't outweigh "service calls" due to scale.

### 2.2 Persona Definition
Clusters shall be profiled based on the **Centroid Vector**. 
*   *Persona A (High Value):* Above-average `Intl_Mins` and `Total_Charge`.
*   *Persona B (Risk Group):* Above-average `CustServ_Calls` and `Day_Mins`.

---

## 3. Supervised Learning Spec (Churn Prediction)

### 3.1 Imbalance Strategy (SMOTE)
*   **Algorithm:** Synthetic Minority Over-sampling Technique.
*   **Mathematical Basis:** For each sample $x_i$ in the minority class, find $k$ nearest neighbors. Create synthetic sample: $x_{new} = x_i + \lambda(x_{zi} - x_i)$ where $\lambda \in [0,1]$.
*   **Implementation:** Must be applied within a `Pipeline` to ensure it only occurs during `fit()`, never during `predict()` or cross-validation fold evaluation.

### 3.2 Model Configuration (XGBoost)
*   **Objective:** `binary:logistic` (Outputting probabilities).
*   **Evaluation Metric:** `aucpr` (Area Under Precision-Recall Curve), superior for imbalanced data.
*   **Hyperparameter Search Space:**
    *   `max_depth`: [3, 5, 7, 9]
    *   `learning_rate`: [0.01, 0.05, 0.1]
    *   `scale_pos_weight`: $\frac{\text{sum(negative instances)}}{\text{sum(positive instances)}}$

---

## 4. API & Interface Specification

### 4.1 Request Schema (`POST /predict`)
```json
{
  "customer_data": {
    "state": "KS",
    "account_length": 128,
    "intl_plan": "no",
    "voice_mail_plan": "yes",
    "number_vmail_messages": 25,
    "total_day_minutes": 265.1,
    "customer_service_calls": 1
  }
}
```

### 4.2 Response Schema
```json
{
  "prediction": {
    "churn_probability": 0.82,
    "risk_level": "High",
    "recommendation": "High Priority Retention Call",
    "top_features": [
      {"feature": "total_day_minutes", "impact": 0.45},
      {"feature": "customer_service_calls", "impact": 0.32}
    ]
  }
}
```

---

## 5. Testing & Quality Assurance

### 5.1 Unit Tests (`pytest`)
*   **Preprocessing Test:** Verify that `Intl_Plan` is correctly encoded to 1/0.
*   **Shape Test:** Ensure the output feature vector matches the model input dimension (e.g., 68 columns after One-Hot Encoding).

### 5.2 Model Tests (Directional Integrity)
*   **Invariance Test:** Changing the `State` of a customer should not drastically change the churn probability.
*   **Monotonicity Test:** Increasing `Customer_Service_Calls` must (generally) correlate with an increase in `Churn_Probability`.

---

## 6. Deployment & Infrastructure

### 6.1 Docker Environment
*   **Base Image:** `python:3.9-slim`
*   **Dependencies:** `pip install -r requirements.txt`
*   **Port Mapping:** `8000:8000`
*   **Entrypoint:** `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`

### 6.2 Monitoring (Post-Deployment)
*   **Logging:** JSON-formatted logs for structured ingestion (e.g., ELK stack).
*   **Drift Detection:** Capture the `y_prob` distribution every 1,000 requests. Use a Kolmogorov-Smirnov test to compare against the training probability distribution.

