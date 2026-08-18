import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_model():
    print("Loading processed data...")
    processed_path = "data/processed/telco_clean.csv"
    
    if not os.path.exists(processed_path):
        print("Processed data not found! Please run etl.py first.")
        return
        
    df = pd.read_csv(processed_path)
    
    # Target variable conversion ('Yes'/'No' to 1/0)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Convert categorical text columns to dummy/numeric variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Accuracy: {acc * 100:.2f}%")
    
    return model

if __name__ == "__main__":
    train_model()