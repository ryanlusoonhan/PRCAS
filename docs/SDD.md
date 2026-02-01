# Software Design Document (SDD): Churn-Shield AI System

**Project Name:** Churn-Shield AI  
**Lead Engineer:** [Your Name]  
**Technology Stack:** Python, FastAPI, XGBoost, Scikit-Learn, Docker, MLflow, DVC  
**Version:** 1.0  

---

## 1. System Architecture
The system follows a **Modular Monolith** architecture, containerized using Docker. It separates the "Training Pipeline" (Asynchronous/Batch) from the "Inference Service" (Synchronous/Real-time).

### 1.1 High-Level Diagram
```text
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

---

## 2. Data Design

### 2.1 Data Schema & Validation
Inputs are validated against a strict schema using **Pandera**.
*   **Feature Set:** 19 independent variables (Categorical: `State`, `Intl_Plan`, etc.; Numerical: `Day_Mins`, `CustServ_Calls`).
*   **Target:** Binary `Churn` (0, 1).
*   **Validation Strategy:** 
    *   Null check: Ensure zero nulls in usage columns.
    *   Range check: `Total_Day_Charge` must be between 0 and 100.

### 2.2 Data Versioning (DVC)
*   **Remote Storage:** Local or S3-compatible storage.
*   **Tracking:** `.dvc` files track hashes of the dataset to ensure that the model trained in `v1.2` can be perfectly reproduced.

---

## 3. Component Design

### 3.1 Data Module (`src/data/`)
*   **Ingestor:** Loads data from DVC-tracked paths.
*   **Validator:** Runs Pandera/Great Expectations suites.
*   **Splitter:** Implements Stratified Shuffling to maintain churn ratios (80/20).

### 3.2 Transformation Module (`src/features/`)
To prevent **Data Leakage**, all transformations are wrapped in a **Scikit-Learn Pipeline**:
*   **Numeric Transformer:** `SimpleImputer` $\rightarrow$ `StandardScaler`.
*   **Categorical Transformer:** `OneHotEncoder(handle_unknown='ignore')`.
*   **Resampler:** Integrated `SMOTE` (via `imblearn.pipeline`) to balance the minority class during the training fit **only**.

### 3.3 Modeling Module (`src/models/`)
*   **Trainer Class:** Orchestrates GridSearch/Optuna.
*   **Experiment Logger:** Wraps the training loop in `mlflow.start_run()` to log:
    *   Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
    *   Artifacts: Feature Importance plots, Confusion Matrix, and the Pickled Pipeline.
*   **Explainer:** Generates `shap.TreeExplainer` objects for every production-ready model.

### 3.4 API Module (`src/api/`)
*   **Framework:** FastAPI.
*   **Endpoints:**
    *   `POST /predict`: Receives JSON, returns churn probability.
    *   `POST /explain`: Returns SHAP values for a specific record.
    *   `GET /health`: Returns system status and model version.

---

## 4. MLOps & DevOps Integration

### 4.1 Containerization (Docker)
A multi-stage `Dockerfile` is used to optimize image size:
1.  **Stage 1 (Build):** Installs dependencies and runs `pytest`.
2.  **Stage 2 (Runtime):** Copies only the necessary code and the `.pkl` model artifact to the final image.

### 4.2 CI/CD Pipeline (GitHub Actions)
*   **On Push:** Trigger Linter (Black/Flake8) and Unit Tests.
*   **On Tag:** Build Docker Image and push to Registry (DockerHub/ECR).
*   **Model Audit:** A script runs to compare the new model's F1-score against the "Champion" model currently in production.

---

## 5. Design Rationale

| Decision | Rationale |
| :--- | :--- |
| **XGBoost** | Superior handling of tabular data and built-in support for missing values and feature importance. |
| **FastAPI** | Asynchronous support allows the system to handle multiple inference requests concurrently without blocking. |
| **SMOTE** | Essential for the 14% minority class to ensure the model doesn't just predict "No Churn" for every customer. |
| **SHAP** | Provides mathematically sound local interpretability, which is a requirement for the "What-If" simulator. |

---

## 6. Performance & Security
*   **Concurrency:** FastAPI workers managed by `uvicorn` to handle up to 500 requests/sec.
*   **Input Sanitization:** Pydantic models in FastAPI enforce type-safety, preventing SQL injection or malformed data attacks.
*   **Logging:** Centralized logging using the Python `logging` library, capturing all 4xx and 5xx errors for debugging.

---

## 7. Future Scalability
*   **Model Monitoring:** Integration with **EvidentlyAI** to track Prediction Drift.
*   **Database:** Transition from CSV to **PostgreSQL** or a Feature Store (e.g., **Feast**) for real-time feature retrieval.
*   **Orchestration:** Moving the training module into **Kubeflow** or **Airflow** for complex scheduling.

