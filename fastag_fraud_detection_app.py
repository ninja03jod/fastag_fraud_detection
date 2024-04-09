import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(r"C:\Users\admin\PycharmProjects\python_Project\MLPROJECTS2\Fastag_Fraud_New_Data.csv")

# Load the encoder
with open(r"C:\Users\admin\PycharmProjects\python_Project\MLPROJECTS2\updated_encode_fastag_data.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the trained models
with open(r"C:\Users\admin\PycharmProjects\python_Project\MLPROJECTS2\fastag_xgb_pipe.pkl", "rb") as f:
    model = pickle.load(f)

# Title of the web app
st.title("Fastag Fraud Detection")

# Sidebar for user input
st.sidebar.header("Input Parameters")

# Dropdowns for selecting input parameters
hour = st.sidebar.slider("Hour", min_value=0, max_value=23, value=12)
month = st.sidebar.slider("Month", min_value=1, max_value=12, value=6)
day_of_week = st.sidebar.selectbox("Day of Week", df['DayOfWeek'].unique())
vehicle_type = st.sidebar.selectbox("Vehicle Type", df['Vehicle_Type'].unique())
fastag_id = st.sidebar.selectbox("Fastag ID", df['FastagID'].unique())
tollbooth_id = st.sidebar.selectbox("Toll Booth ID", df['TollBoothID'].unique())
lane_type = st.sidebar.selectbox("Lane Type", df['Lane_Type'].unique())
vehicle_dimensions = st.sidebar.selectbox("Vehicle Dimensions", df['Vehicle_Dimensions'].unique())
geographical_location = st.sidebar.selectbox("Geographical Location", df['Geographical_Location'].unique())
transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
amount_paid = st.sidebar.number_input("Amount Paid", min_value=0.0)
vehicle_speed = st.sidebar.number_input("Vehicle Speed", min_value=0.0)

user_inputs = pd.DataFrame({
    'Hour': [hour],
    'Month': [month],
    'DayOfWeek': [day_of_week],
    'Vehicle_Type': [vehicle_type],
    'FastagID': [fastag_id],
    'TollBoothID': [tollbooth_id],
    'Lane_Type': [lane_type],
    'Vehicle_Dimensions': [vehicle_dimensions],
    'Geographical_Location': [geographical_location],
    'Transaction_Amount': [transaction_amount],
    'Amount_Paid': [amount_paid],
    'Vehicle_Speed': [vehicle_speed]
})

for col in ['Vehicle_Type','FastagID','TollBoothID','Lane_Type','Vehicle_Dimensions','Geographical_Location']:
    user_inputs[col] = encoder.fit_transform(user_inputs[col])

# Add default value for 'Amount_paid' if missing
if 'Amount_paid' not in user_inputs.columns:
    user_inputs['Amount_paid'] = 0.0  # You can choose an appropriate default value

try:
    prediction = model.predict(user_inputs)
    st.subheader("Prediction:")
    if prediction == 0:
        st.write("Fraud")
    else:
        st.write("Not Fraud")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.subheader("Dataset:")
st.write(df)
