import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(page_title="Telco Churn Prediction Dashboard", layout="wide")

st.title("📞 Telco Customer Churn Intelligence System")
st.markdown("Predict whether a customer is likely to cancel their service based on their profile and account details.")

# Load and cache data/model training
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("data/processed/telco_clean.csv")
    
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X.columns

model, feature_columns = load_and_train_model()

# Sidebar User Inputs
st.sidebar.header("Customer Parameters")

tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0)
total_charges = st.sidebar.slider("Total Charges ($)", min_value=0.0, max_value=9000.0, value=850.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Main Panel Display
col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Customer Profile")
    st.write(f"* **Tenure:** {tenure} months")
    st.write(f"* **Monthly Charges:** ${monthly_charges:.2f}")
    st.write(f"* **Total Charges:** ${total_charges:.2f}")
    st.write(f"* **Contract:** {contract}")
    st.write(f"* **Internet Service:** {internet_service}")
    st.write(f"* **Payment Method:** {payment_method}")

# Build prediction input dataframe matching training columns
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

if 'tenure' in input_data.columns: input_data['tenure'] = tenure
if 'MonthlyCharges' in input_data.columns: input_data['MonthlyCharges'] = monthly_charges
if 'TotalCharges' in input_data.columns: input_data['TotalCharges'] = total_charges

# Handle categorical one-hot mappings dynamically if columns exist
for col in input_data.columns:
    if contract in col and col in input_data.columns: input_data[col] = 1
    if internet_service in col and col in input_data.columns: input_data[col] = 1
    if payment_method in col and col in input_data.columns: input_data[col] = 1

with col2:
    st.subheader("Churn Prediction Result")
    if st.button("Predict Churn Risk", type="primary"):
        prediction_prob = model.predict_proba(input_data)[0][1]
        
        if prediction_prob > 0.5:
            st.error(f"🚨 **High Risk of Churn!** (Probability: {prediction_prob * 100:.1f}%)")
            st.markdown("*Recommended Action:* Offer this customer a loyalty discount or promote a 1-year contract extension.")
        else:
            st.success(f"✅ **Low Risk of Churn.** (Probability: {prediction_prob * 100:.1f}%)")
            st.markdown("*Recommended Action:* Customer is stable. Standard account maintenance.")