# NYC Taxi Trip Duration Prediction 🚖

This project is part of the **CAPABL Data Science Project**. It focuses on predicting the total duration of taxi trips in New York City based on various features such as pickup/dropoff locations, passenger count, and time of travel.

## 📊 Live Demo

Check out the interactive web application here:
**[Predict Taxi Trip Duration](https://predicttaxitripduration.streamlit.app/)**

---

## 🚀 Project Overview

The goal of this project is to build a regression model that can accurately estimate how long a taxi trip will take. This is a common challenge in urban mobility and logistics, where precise timing helps optimize fleet management and passenger expectations.

### Key Features of the App:

* **Interactive Inputs:** Users can manually enter latitude/longitude coordinates or use a map interface.
* **Distance Calculation:** Automatically calculates the geodesic distance between coordinates using the `geopy` library.
* **Real-time Prediction:** Uses a trained Machine Learning model to provide an estimate in seconds/minutes.

---

## 🛠️ Technology Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Ridge, Lasso, Linear Regression)
* **Web Framework:** Streamlit (for the Live Demo)
* **Deployment:** Streamlit Cloud & GitHub

---

## 📂 Dataset Description

The dataset used for this project contains over **1.4 million** records of taxi trips. Key columns include:

* `vendor_id`: Code indicating the provider associated with the trip record.
* `pickup_datetime` & `dropoff_datetime`: Timestamps for the start and end of the trip.
* `passenger_count`: Number of passengers in the vehicle.
* `pickup_longitude/latitude`: Geographic coordinates where the meter was engaged.
* `dropoff_longitude/latitude`: Geographic coordinates where the meter was disengaged.
* `trip_duration`: The target variable, duration of the trip in seconds.

---

## 📈 Model Performance

Multiple regression models were evaluated to find the best fit:
| Model | R² Score |
| :--- | :--- |
| **Linear Regression** | 0.0289 |
| **Lasso Regression** | 0.0289 |
| **Ridge Regression** | 0.0289 |

While the initial scores are low, further feature engineering (such as adding weather data or holiday indicators) can be explored to improve accuracy.

---

## 💻 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/Mousoomi03/predict_taxi_trip_duration.git
cd predict_taxi_trip_duration

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the app:**
```bash
streamlit run app.py

```
