import pandas as pd
import os

def run_etl():
    print("Starting ETL Pipeline...")
    
    # URL for the classic Telco Churn dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    # Define paths
    raw_path = "data/raw/telco_raw.csv"
    processed_path = "data/processed/telco_clean.csv"
    
    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Extract
    print("Downloading dataset...")
    df = pd.read_csv(url)
    df.to_csv(raw_path, index=False)
    
    # 2. Transform
    print("Cleaning data...")
    # TotalCharges has some blank spaces for new customers; convert to numeric and fill NaNs
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID as it's a unique identifier, not a feature
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # 3. Load
    print(f"Saving processed data to {processed_path}...")
    df.to_csv(processed_path, index=False)
    print("ETL Pipeline completed successfully!")

if __name__ == "__main__":
    run_etl()