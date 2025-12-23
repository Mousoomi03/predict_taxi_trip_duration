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
    # 1. Calculate the 'dis' feature using geopy
    dist = geopy.distance.geodesic((p_lat, p_lon), (d_lat, d_lon)).km
    
    # 2. Create the exact 13 columns your model learned in Colab
    # We will use some default values for time for now to test it
    input_dict = {
        'vendor_id': [1],
        'passenger_count': [passengers],
        'pickup_longitude': [p_lon],
        'pickup_latitude': [p_lat],
        'dropoff_longitude': [d_lon],
        'dropoff_latitude': [d_lat],
        'dis': [dist],
        'month': [1],         # January (encoded as 1)
        'pickup_day': [0],    # Monday (encoded as 0)
        'pickup_hour': [12],   # Noon
        'pickup_min': [30],
        'store_and_fwd_flag_N': [1], # Most trips are 'N'
        'store_and_fwd_flag_Y': [0]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_dict)
    
    # 3. Make the prediction
    prediction = model.predict(input_df)
    
    # Display the result
    st.success(f"✅ Estimated Trip Duration: {int(prediction[0]/60)} Minutes")
    st.write(f"Calculated Distance: {round(dist, 2)} km")
