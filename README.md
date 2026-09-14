
# Credit Card Fraud Detection & Risk Monitoring System

A prototype fraud detection and risk monitoring system demonstrating how a machine-learning model can be integrated into a payment-processing workflow.

The system uses a Random Forest Classifier to predict potentially fraudulent credit card transactions. It provides a FastAPI REST API for transaction predictions, a risk engine for recommending APPROVE, REVIEW, or BLOCK actions, an SQLite database for storing prediction results, and a Streamlit dashboard for monitoring transactions.

---

## Project Overview

Credit card fraud detection is an important machine-learning application because fraudulent transactions are usually rare compared with normal transactions.

This project demonstrates an end-to-end fraud detection workflow:

```text
Transaction Data
       |
       v
FastAPI REST API
       |
       v
Input Validation
       |
       v
Machine Learning Model
(Random Forest Classifier)
       |
       v
Fraud Probability
       |
       v
Risk Engine
       |
       v
Risk Level and Recommended Action
       |
       v
SQLite Database
       |
       v
Streamlit Monitoring Dashboard
```

The project is designed as an educational and demonstration prototype. It is not intended to be used as a real banking or production payment-processing system.

---

## Main Features

- Credit card fraud classification using machine learning
- Random Forest model training
- Logistic Regression and Random Forest model comparison
- Fraud probability prediction
- FastAPI REST API
- Interactive Swagger API documentation
- Risk classification into LOW, MEDIUM, and HIGH
- Recommended actions:
  - APPROVE
  - REVIEW
  - BLOCK
- SQLite database storage
- Streamlit fraud monitoring dashboard
- Recent transaction history
- Fraud-alert monitoring
- Automated testing using Pytest
- Demo transaction testing
- Model performance evaluation
- Confusion matrix and class distribution visualization

---

## Dataset

This project uses the ULB Credit Card Fraud Detection dataset.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Information

The dataset contains credit card transactions with the following main columns:

| Column | Description |
|---|---|
| `Time` | Number of seconds elapsed between the first transaction and the current transaction |
| `V1` to `V28` | Anonymized PCA-transformed numerical features |
| `Amount` | Transaction amount |
| `Class` | Target label |

### Target Variable

The `Class` column is the target variable:

```text
0 = Normal transaction
1 = Fraudulent transaction
```

### Anonymized Features

The features `V1` to `V28` are anonymized principal components generated using Principal Component Analysis.

These features represent mathematical combinations of confidential transaction information. Their exact real-world meanings are not disclosed in the dataset.

The machine-learning model uses patterns from these features, together with `Time` and `Amount`, to classify transactions.

### Dataset Characteristics

- Highly imbalanced dataset
- Normal transactions are much more common than fraudulent transactions
- Fraud detection requires careful evaluation of precision, recall, F1-score, and ROC-AUC
- The original dataset should not be uploaded to GitHub because it is large

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning preprocessing, training, and evaluation |
| Random Forest | Main fraud classification model |
| Logistic Regression | Baseline model for comparison |
| FastAPI | REST API development |
| Uvicorn | Running the FastAPI application |
| Streamlit | Interactive monitoring dashboard |
| SQLite | Storing transaction prediction records |
| Pytest | Automated testing |
| Joblib | Saving and loading trained models |
| Matplotlib | Data visualization |
| Seaborn | Class distribution and evaluation visualizations |
| Git/GitHub | Version control and project hosting |

---

## System Architecture

### 1. Transaction Input

A transaction contains:

- `Time`
- `V1` to `V28`
- `Amount`

The `Class` column is used during model training but is not required when making a prediction.

### 2. FastAPI REST API

The transaction is sent to the FastAPI backend using a `POST /predict` request.

The API validates the input and passes the transaction features to the trained model.

### 3. Machine-Learning Prediction

The Random Forest model produces:

- Transaction prediction
- Fraud probability

Example:

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.95
}
```

### 4. Risk Engine

The risk engine converts the fraud probability into a risk level and recommended action.

The demonstration thresholds are:

| Fraud Probability | Risk Level | Recommended Action |
|---|---|---|
| Less than `0.30` | LOW | APPROVE |
| `0.30` to less than `0.70` | MEDIUM | REVIEW |
| `0.70` or higher | HIGH | BLOCK |

These thresholds are demonstration rules and are not real banking risk policies.

### 5. Database Storage

The prediction result is stored in an SQLite database.

The database stores information such as:

- Transaction ID
- Transaction timestamp
- Transaction amount
- Fraud probability
- Prediction
- Risk level
- Recommended action

### 6. Streamlit Dashboard

The dashboard displays:

- Total transactions
- Fraud alerts
- Normal transactions
- High-risk transactions
- Fraud probability charts
- Risk-level information
- Recent transaction records
- Demo transaction testing

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── features.pkl
│
├── src/
│   ├── train.py
│   ├── model_service.py
│   └── risk_engine.py
│
├── tests/
│   └── test_fraud_pipeline.py
│
├── Images/
│   ├── model_training.png
│   ├── api_health_check.png
│   ├── normal_transaction.png
│   ├── fraud_transaction.png
│   ├── monitoring_dashboard.png
│   └── automated_tests.png
│
├── app.py
├── create_demo.py
├── database.py
├── generate_demo_payloads.py
├── requirements.txt
├── README.md
└── .gitignore
```

> Note: The dataset, database, virtual environment, and generated transaction files should not be uploaded to a public GitHub repository.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
```

Move into the project directory:

```bash
cd credit-card-fraud-detection
```

### 2. Create a Virtual Environment

For Windows:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

For macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded file in the following location:

```text
data/creditcard.csv
```

The expected file path is:

```text
data/creditcard.csv
```

Do not upload the dataset to GitHub if the repository is public.

---

## Train the Machine-Learning Model

Run the training script from the project root:

```powershell
python src/train.py
```

The training process performs tasks such as:

1. Loading the dataset
2. Inspecting the class distribution
3. Preparing the input features
4. Handling the imbalanced dataset
5. Splitting the data into training and testing sets
6. Training Logistic Regression
7. Training Random Forest
8. Comparing model performance
9. Evaluating precision, recall, F1-score, and ROC-AUC
10. Generating a confusion matrix
11. Saving the selected model

The trained model files are saved in the `models/` directory.

Example model files:

```text
models/fraud_model.pkl
models/features.pkl
```

---

## Model Performance

The Random Forest model was selected as the production model for this prototype.

Example evaluation results:

| Metric | Random Forest |
|---|---:|
| Precision | 0.9740 |
| Recall | 0.7895 |
| F1-score | 0.8721 |
| ROC-AUC | 0.9690 |

### Confusion Matrix

```text
[[1998,    2],
 [  20,   75]]
```

The model evaluation shows that the Random Forest model can identify many fraudulent transactions while maintaining high precision.

The results are based on the current experimental dataset preparation and should not be interpreted as production banking performance.

---

## Run the FastAPI Backend

Start the FastAPI server from the project root:

```powershell
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Swagger API Documentation

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface allows you to test the available API endpoints.

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Fraud Prediction

```http
POST /predict
```

The request should contain:

- `Time`
- `V1` to `V28`
- `Amount`

The `Class` column must not be included in the prediction request.

Example response for a normal transaction:

```json
{
  "prediction": "NORMAL",
  "fraud_probability": 0.01,
  "risk_level": "LOW",
  "recommended_action": "APPROVE"
}
```

Example response for a fraudulent transaction:

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.95,
  "risk_level": "HIGH",
  "recommended_action": "BLOCK"
}
```

---

## Run the Streamlit Dashboard

Open a second terminal and activate the virtual environment.

Run:

```powershell
streamlit run app.py
```

The dashboard will open in your browser.

The dashboard provides:

- Transaction statistics
- Fraud prediction information
- Risk-level analysis
- Recent transaction records
- Fraud probability charts
- Demo transaction testing

---

## Demo Transactions

The project includes scripts for creating and generating demo transactions.

### Create Demo Transactions

```powershell
python create_demo.py
```

This generates a demo transaction file:

```text
demo_transactions.csv
```

### Generate JSON Payloads

```powershell
python generate_demo_payloads.py
```

This generates:

```text
normal_transaction.json
fraud_transaction.json
```

These files can be used to test the FastAPI prediction endpoint.

For a public repository, generated transaction files should normally be excluded using `.gitignore`.

---

## Run Automated Tests

Run the test suite from the project root:

```powershell
python -m pytest
```

Example result:

```text
9 passed, 2 warnings
```

The tests validate important parts of the fraud detection pipeline, including:

- API functionality
- Model prediction behavior
- Risk classification
- Database operations
- Input validation
- Fraud pipeline components

The warnings shown during testing are dependency deprecation warnings and do not indicate test failures.

---

## Example Payment-Processing Workflow

A simplified real-world payment workflow can be represented as follows:

```text
Customer makes an online card payment
              |
              v
Payment gateway sends transaction details
              |
              v
FastAPI receives the transaction
              |
              v
Random Forest model predicts fraud probability
              |
              v
Risk engine assigns LOW, MEDIUM, or HIGH risk
              |
              v
System recommends APPROVE, REVIEW, or BLOCK
              |
              v
Prediction is stored in SQLite
              |
              v
Transaction appears on Streamlit dashboard
```

### Example 1: Normal Transaction

```text
Transaction Amount: $25.00
Fraud Probability: 0.04
Risk Level: LOW
Recommended Action: APPROVE
```

### Example 2: Suspicious Transaction

```text
Transaction Amount: $2,000.00
Fraud Probability: 0.55
Risk Level: MEDIUM
Recommended Action: REVIEW
```

### Example 3: High-Risk Transaction

```text
Transaction Amount: $1,200.00
Fraud Probability: 0.91
Risk Level: HIGH
Recommended Action: BLOCK
```

These examples are illustrative demonstrations of the risk engine and do not represent actual banking decisions.

---

## Screenshots

### Model Training Results

The training output compares Logistic Regression and Random Forest models using precision, recall, F1-score, ROC-AUC, and a confusion matrix.

![Model Training Results](Images/model_training.png)

### FastAPI Health Check

The FastAPI Swagger interface confirms that the API is healthy and the trained model has been loaded.

![FastAPI Health Check](Images/api_health_check.png)

### Normal Transaction Approved

The API classifies a normal transaction as NORMAL with a LOW risk level and APPROVE recommendation.

![Normal Transaction Approved](Images/normal_transaction.png)

### Fraudulent Transaction Blocked

The API classifies a fraudulent transaction as FRAUD with a HIGH risk level and BLOCK recommendation.

![Fraudulent Transaction Blocked](Images/fraud_transaction.png)

### Fraud Monitoring Dashboard

The Streamlit dashboard displays recent transactions, fraud probabilities, predictions, risk levels, and recommended actions.

![Fraud Monitoring Dashboard](Images/monitoring_dashboard.png)

### Automated Tests

The Pytest output confirms that the automated test suite passed successfully.

![Automated Tests](Images/automated_tests.png)

---

## Security and Privacy Notes

Do not upload the following files to a public GitHub repository:

```text
venv/
.venv/
data/creditcard.csv
fraud_detection.db
*.db
normal_transaction.json
fraud_transaction.json
demo_transactions.csv
.env
__pycache__/
.pytest_cache/
```

The dataset contains transaction-related information and should be handled carefully.

Never upload:

- Passwords
- API keys
- Access tokens
- Personal customer information
- Private banking information
- Sensitive configuration files

---

## Limitations

This project has several limitations:

1. The dataset uses anonymized PCA features.
2. The original business meaning of `V1` to `V28` is not available.
3. The system does not use real customer identity information.
4. The system does not include live payment gateway integration.
5. The risk thresholds are demonstration thresholds.
6. The model is not trained for a specific bank or payment provider.
7. The model should not be used directly for real financial decisions.
8. Production fraud detection would require additional information such as:
   - Customer transaction history
   - Device information
   - IP address
   - Location
   - Merchant details
   - Authentication information
   - Transaction frequency
   - Previous fraud behavior
   - Real-time model monitoring

---

## Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Cross-validation
- Advanced imbalance-handling methods
- SMOTE or other resampling techniques
- XGBoost or other advanced classifiers
- Real-time transaction streaming
- User authentication
- Role-based dashboard access
- Email or SMS fraud alerts
- Model explainability using SHAP
- Model monitoring and drift detection
- Docker deployment
- Cloud deployment
- CI/CD pipeline
- Integration with a payment gateway sandbox
- More advanced fraud-risk rules
- Real-time analytics
- Improved database design

---

## Learning Outcomes

This project demonstrates practical experience in:

- Data preprocessing
- Exploratory data analysis
- Imbalanced classification
- Machine-learning model training
- Model comparison
- Model evaluation
- REST API development
- Backend integration
- Risk-engine design
- Database storage
- Dashboard development
- Automated testing
- Git and GitHub workflow
- End-to-end application development

---

# 👤 Author

**Haris**

Final Year Undergraduate

Department of Electrical and Electronic Engineering

University of Jaffna
