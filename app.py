import streamlit as st
import pandas as pd
import joblib
import geopy.distance

# Load the model
model = joblib.load('taxi_model.pkl')

st.title("🚖 NYC Taxi Duration Predictor")

# Input fields
p_lat = st.number_input("Pickup Latitude", value=40.71)
p_lon = st.number_input("Pickup Longitude", value=-74.00)
d_lat = st.number_input("Dropoff Latitude", value=40.73)
d_lon = st.number_input("Dropoff Longitude", value=-73.99)
passengers = st.slider("Passengers", 1, 6, 2)

if st.button("Predict"):
    # 1. Feature Engineering
    dist = geopy.distance.geodesic((p_lat, p_lon), (d_lat, d_lon)).km
    
    # 2. Build the DataFrame with EXACT names used in training
    # Note: I used generic values for time/vendor to ensure it runs first
    input_dict = {
        'vendor_id': [1],
        'passenger_count': [passengers],
        'pickup_longitude': [p_lon],
        'pickup_latitude': [p_lat],
        'dropoff_longitude': [d_lon],
        'dropoff_latitude': [d_lat],
        'dis': [dist],
        'month': [1],        # Encoded value for Month
        'pickup_day': [1],   # Encoded value for Day
        'pickup_hour': [12],
        'pickup_min': [30],
        'store_and_fwd_flag_N': [1],
        'store_and_fwd_flag_Y': [0]
    }
    
    input_df = pd.DataFrame(input_dict)

    # --- THE MAGIC FIX ---
    # This ensures the order of columns matches the model exactly
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]
    
    # 3. Predict
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Estimated Trip Duration: {int(prediction[0]/60)} Minutes")
        st.write(f"Calculated Distance: {round(dist, 2)} km")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        # This will print the expected names so you can see the mismatch
        st.write("Model expects these columns:", model.feature_names_in_)
