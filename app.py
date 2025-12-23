import streamlit as st
import pandas as pd
import joblib
import geopy.distance

# Load files
model = joblib.load('taxi_model.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🚖 NYC Taxi Duration Predictor")

# Inputs
p_lat = st.number_input("Pickup Latitude", value=40.71)
p_lon = st.number_input("Pickup Longitude", value=-74.00)
d_lat = st.number_input("Dropoff Latitude", value=40.73)
d_lon = st.number_input("Dropoff Longitude", value=-73.99)
passengers = st.slider("Passengers", 1, 6, 1)

if st.button("Predict"):
    # Calculate distance
    dist = geopy.distance.geodesic((p_lat, p_lon), (d_lat, d_lon)).km
    
    # Create the data for the model (simplified for demonstration)
    # Your model expects 13 features. We must provide them in order.
    # [vendor_id, passenger_count, p_lon, p_lat, d_lon, d_lat, dis, month, day, hour, min, flag_N, flag_Y]
    input_data = [[1, passengers, p_lon, p_lat, d_lon, d_lat, dist, 1, 1, 12, 30, 1, 0]]
    
    prediction = model.predict(input_data)
    st.header(f"Trip Duration: {int(prediction[0]/60)} Minutes")
