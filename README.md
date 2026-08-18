# telco-customer-churn-intelligence
An end-to-end data science portfolio project featuring an automated ETL data pipeline, a Random Forest machine learning classification model, and an interactive Streamlit web dashboard for real-time customer churn prediction.

# Telco Customer Churn Intelligence System

An end-to-end data science portfolio project built to predict customer churn risk using an automated ETL pipeline, a machine learning classification model, and an interactive web dashboard.

---

## 📊 Project Overview
Customer churn is a critical challenge for subscription-based businesses. This project provides an automated data solution that ingests raw customer profiles, cleans missing values, trains a predictive model, and offers business stakeholders a real-time tool to evaluate customer retention risk and deploy proactive mitigation strategies.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Manipulation:** Pandas, Scikit-Learn
* **Machine Learning Model:** Random Forest Classifier
* **Web Dashboard UI:** Streamlit

---

## ⚙️ Architecture & Workflow
1. **ETL Pipeline (`src/etl.py`):** Automatically fetches raw customer records, handles missing data values, cleans data types, and outputs processed datasets to `data/processed/`.
2. **Model Training (`src/train.py`):** Encodes categorical features, splits data into training and testing sets, and trains a `RandomForestClassifier` to evaluate predictive performance.
3. **Web Application (`app.py`):** An interactive Streamlit interface allowing users to dynamically input customer parameters (such as monthly charges, tenure, and contract types) to instantly test churn risk and view data-backed recommendations.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
   
Install the required dependencies:
   pip install pandas scikit-learn streamlit
   
Run the ETL pipeline to prepare the dataset:
   python src/etl.py

 Launch the Streamlit dashboard:
   streamlit run app.py


   
