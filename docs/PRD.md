# Product Requirement Document (PRD): Churn-Shield AI Platform

**Project Name:** Churn-Shield AI (Telecommunications Retention Engine)  
**Product Manager/Lead MLE:** [Your Name]  
**Version:** 1.0  
**Date:** October 2023  

---

## 1. Product Overview
Churn-Shield AI is an end-to-end predictive analytics platform that identifies at-risk telecommunications customers. Unlike standard churn scripts, this product provides **prescriptive insights**, allowing users to perform "What-If" simulations to determine which retention lever (e.g., offering a discount, reducing service lag) will most effectively prevent a specific customer from leaving.

---

## 2. User Personas
| Persona | User Goal | Product Interaction |
| :--- | :--- | :--- |
| **Operational CSR** (Customer Service Rep) | Retain a customer currently on the phone. | Uses the Web UI to see the customer’s "Risk Score" and "Top Churn Drivers" in real-time. |
| **Marketing Analyst** | Design high-level retention campaigns. | Uses the "Segment Explorer" to find which clusters respond best to specific plan types. |
| **MLE / Admin** | Maintain system health. | Monitors model performance, re-trains models, and checks for data drift via the API dashboard. |

---

## 3. User Stories
*   **As a CSR**, I want to see the SHAP (explainability) values for a customer so that I know exactly what complaint to address to keep them.
*   **As an Analyst**, I want to export a list of the "Top 5% Highest Risk" customers into a CSV for an email blast.
*   **As an MLE**, I want the system to reject data that doesn't match the training schema so the model doesn't produce "garbage" predictions.

---

## 4. Feature Specifications

### 4.1 Data Ingestion & Validation (The "Robustness" Feature)
*   **Feature:** Automated Schema Enforcement.
*   **Specification:** Every input (via CSV or API) must be validated using **Pandera**. 
*   **Validation Rules:** `Account_Length` > 0; `Total_Day_Calls` < 500; `Churn` is Boolean.
*   **Error Handling:** Invalid records are logged to a `rejects/` folder with a timestamp and reason.

### 4.2 The Segmentation Engine (Unsupervised)
*   **Feature:** Persona Clustering.
*   **Specification:** Use K-Means (optimized via Silhouette Score). 
*   **Output:** Every customer ID is tagged with a `Segment_ID`. 
*   **Visual Requirement:** A 2D PCA/t-SNE plot in the UI showing the current customer’s position relative to their cluster.

### 4.3 The Prediction Engine (Supervised)
*   **Feature:** Probability Inference.
*   **Specification:** XGBoost model with SMOTE-balanced training.
*   **Performance Requirement:** F1-score $\geq$ 0.85 on test set.
*   **Output:** A Churn Probability (e.g., 0.82) and a Risk Level (Low, Medium, High).

### 4.4 Prescription Tool (The "0.1% World-Class" Feature)
*   **Feature:** "What-If" Simulator.
*   **Specification:** Within the UI, allow the user to modify feature values (e.g., simulate a customer receiving 2 fewer service calls) and re-run the prediction.
*   **Goal:** Calculate the "Churn Reduction Potential" for specific interventions.

---

## 5. Technical System Design

### 5.1 Architecture
*   **Backend:** FastAPI for high-performance RESTful endpoints.
*   **Frontend:** Streamlit for a rapid, interactive data dashboard.
*   **Orchestration:** DVC (Data Version Control) for tracking data changes.
*   **Environment:** Dockerized (multi-container: `api`, `dashboard`, `mlflow`).

### 5.2 The Inference Pipeline
1.  **Request:** User inputs data.
2.  **Preprocessing:** Data is scaled/encoded using saved `Joblib` objects.
3.  **Model:** XGBoost generates probability.
4.  **Explainability:** SHAP generates feature contributions.
5.  **Response:** JSON payload containing: `score`, `risk_level`, `shap_values`, `persona`.

---

## 6. Model Governance & Maintenance
*   **Versioning:** Every model iteration is tagged in **MLflow** (e.g., `v1.0.1-production`).
*   **Monitoring:** The system logs the distribution of incoming "Daily Minutes." If the mean shifts by more than 2 standard deviations, the system flags a "Data Drift" warning for the MLE.
*   **Retraining Logic:** The PRD assumes a monthly batch re-training cycle triggered by GitHub Actions.

---

## 7. Acceptance Criteria (Definition of Done)
*   [ ] Model exceeds 0.80 Recall and 0.85 Precision on unseen test data.
*   [ ] API passes a load test of 50 requests per second.
*   [ ] All code passes `flake8` linting and `pytest` unit tests.
*   [ ] SHAP explanations align with documented domain knowledge (e.g., high service calls correlate with churn).

