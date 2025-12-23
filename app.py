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
    # 1. Feature Engineering (Calculate distance)
    dist = geopy.distance.geodesic((p_lat, p_lon), (d_lat, d_lon)).km
    
    # 2. Build the exact 12 columns used in your Colab 'X' dataframe
    input_dict = {
        'vendor_id': [1],
        'passenger_count': [passengers],
        'pickup_longitude': [p_lon],
        'pickup_latitude': [p_lat],
        'dropoff_longitude': [d_lon],
        'dropoff_latitude': [d_lat],
        'store_and_fwd_flag': [0], # 0 for 'N', 1 for 'Y'
        'dis': [dist],
        'month': [3],               # March (approximate)
        'pickup_day': [1],          # Monday (approximate)
        'pickup_hour': [12],
        'pickup_min': [30]
    }
    
    input_df = pd.DataFrame(input_dict)

    # 3. Match the order of columns exactly with X_train
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df[model.feature_names_in_]
    
    # 4. Make the prediction
    try:
        prediction = model.predict(input_df)
        # Prediction is in seconds, converting to minutes
        st.success(f"✅ Estimated Trip Duration: {int(prediction[0]/60)} Minutes")
        st.info(f"Calculated Trip Distance: {round(dist, 2)} km")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
