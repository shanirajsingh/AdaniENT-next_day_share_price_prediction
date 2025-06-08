import streamlit as st
import numpy as np
import joblib  # for loading saved model

# Load your trained model
open_best_model = joblib.load("openprice_model.pkl")  # Replace with your model file name
high_best_model = joblib.load("highprice_model.pkl")
low_best_model = joblib.load("lowprice_model.pkl")
close_best_model = joblib.load("closeprice_model.pkl")

def predict_tomorrow_open(open_price, high_price, low_price, close_price):
    input_data = np.array([[open_price, high_price, low_price, close_price]])
    open_prediction = open_best_model.predict(input_data)
    high_prediction = high_best_model.predict(input_data)
    low_prediction = low_best_model.predict(input_data)
    close_prediction = close_best_model.predict(input_data)
    return open_prediction[0], high_prediction[0], low_prediction[0], close_prediction[0]

# Streamlit UI
st.set_page_config(page_title="ADANIENT Next day Price Predictor")
st.title("📈 Adani Enterprises Limited (ADANIENT.NS) Next day Price Predictor")
st.write("Enter today's Open, High, Low, and Close prices to predict tomorrow's  price.")

# Input fields
open_price = st.number_input("Today’s Open Price", format="%.2f")
high_price = st.number_input("Today’s High Price", format="%.2f")
low_price = st.number_input("Today’s Low Price", format="%.2f")
close_price = st.number_input("Today’s Close Price", format="%.2f")

# Predict button
if st.button("Predict"):
    result = predict_tomorrow_open(open_price, high_price, low_price, close_price)
    st.write("Loading...")
    st.success(f"📊 Predicted Tomorrow's Open Price: ₹{result[0]}")
    st.success(f"📊 Predicted Tomorrow's high Price: ₹{result[1]}")
    st.success(f"📊 Predicted Tomorrow's low Price: ₹{result[2]}")
    st.success(f"📊 Predicted Tomorrow's close Price: ₹{result[3]}")

  
    

#streamlit run app.py
