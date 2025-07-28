"""import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Solar Power Energy Prediction Dashboard")
st.write("This dashboard will help you visualize and predict solar power energy production based on various parameters.")   
st.sidebar.header("Input Parameters")

# Load and parse data
df = pd.read_csv("Merged Data.csv", parse_dates=["timestamp"])

# Select relevant columns
dfr = df[["timestamp", "P"]].copy()

# Display raw data
st.write("### Data Preview")
st.dataframe(dfr)

# Interactive time series chart using Plotly
st.write("### Power Over Time (Interactive)")
fig = px.line(dfr, x="timestamp", y="P", title="Solar Power Production Over Time", labels={"P": "Power (W)", "timestamp": "Time"})
st.plotly_chart(fig, use_container_width=True)

import prophet 

prophet.future"""



import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib

# Page configuration
st.set_page_config(
    page_title="Solar Energy Forecast",
    page_icon="☀️",
    layout="wide"
)

# Load model and features
@st.cache_resource
def load_model_and_features():
    """Load the trained model and feature names"""
    try:
        # Load model (assuming it's a joblib/pickle file despite .txt extension)
        with open('solar_energy_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        
        # Load feature names
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_weather_forecast(lat=40.7128, lon=-74.0060):
    """
    Fetch 7-day weather forecast from Open-Meteo API
    Default coordinates are for New York City
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m", 
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation"
        ],
        "forecast_days": 7,
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def process_weather_data(weather_data, feature_names):
    """Convert weather API response to model input format"""
    if not weather_data:
        return None
        
    hourly = weather_data['hourly']
    
    # Create DataFrame from weather data
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'temperature': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'precipitation': hourly['precipitation'],
        'cloud_cover': hourly['cloud_cover'],
        'wind_speed': hourly['wind_speed_10m'],
        'shortwave_radiation': hourly['shortwave_radiation'],
        'direct_radiation': hourly['direct_radiation'],
        'diffuse_radiation': hourly['diffuse_radiation']
    })
    
    # Add time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Create a feature matrix matching your model's expected features
    # This is a basic mapping - you may need to adjust based on your actual features
    feature_mapping = {
        'temperature_2m': 'temperature',
        'relative_humidity_2m': 'humidity', 
        'precipitation': 'precipitation',
        'cloud_cover': 'cloud_cover',
        'wind_speed_10m': 'wind_speed',
        'shortwave_radiation': 'shortwave_radiation',
        'direct_radiation': 'direct_radiation',
        'diffuse_radiation': 'diffuse_radiation',
        'hour': 'hour',
        'day_of_week': 'day_of_week',
        'month': 'month'
    }
    
    # Initialize feature array
    X = np.zeros((len(df), len(feature_names)))
    
    # Map available features
    for i, feature in enumerate(feature_names):
        if feature in feature_mapping and feature_mapping[feature] in df.columns:
            X[:, i] = df[feature_mapping[feature]].values
        elif feature in df.columns:
            X[:, i] = df[feature].values
    
    return X, df['datetime']

def main():
    # Header
    st.title("☀️ Solar Energy Production Forecast")
    st.markdown("7-day solar energy production prediction based on weather forecast")
    
    # Sidebar for location input
    st.sidebar.header("Location Settings")
    lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
    
    # Option 1: Auto-update when location changes
    # Create a unique key based on location to detect changes
    location_key = f"{lat:.4f}_{lon:.4f}"
    
    # Check if location has changed
    location_changed = False
    if 'current_location' not in st.session_state:
        st.session_state.current_location = location_key
        location_changed = True
    elif st.session_state.current_location != location_key:
        st.session_state.current_location = location_key
        location_changed = True
    
    # Update forecast button
    update_clicked = st.sidebar.button("Update Forecast")
    
    # Trigger update if button clicked, location changed, or no data exists
    if update_clicked or location_changed or 'forecast_data' not in st.session_state:
        with st.spinner("Loading model and fetching weather data..."):
            # Load model
            model, feature_names = load_model_and_features()
            
            if model is None:
                st.error("Could not load the solar energy model. Please check your model files.")
                return
            
            # Get weather forecast
            weather_data = get_weather_forecast(lat, lon)
            
            if weather_data is None:
                st.error("Could not fetch weather data. Please try again.")
                return
            
            # Process data for prediction
            X, timestamps = process_weather_data(weather_data, feature_names)
            
            if X is None:
                st.error("Could not process weather data.")
                return
            
            # Make predictions
            try:
                predictions = model.predict(X)
                # Ensure predictions are non-negative (solar energy can't be negative)
                predictions = np.maximum(predictions, 0)
                
                # Store in session state with location info
                st.session_state.forecast_data = {
                    'predictions': predictions,
                    'timestamps': timestamps,
                    'weather_data': weather_data,
                    'location': {'lat': lat, 'lon': lon}
                }
                
                # Show success message when location changes
                if location_changed and not update_clicked:
                    st.success(f"Forecast updated for new location: {lat:.4f}, {lon:.4f}")
                
            except Exception as e:
                st.error(f"Error making predictions: {e}")
                return
    
    # Display forecast if available
    if 'forecast_data' in st.session_state:
        data = st.session_state.forecast_data
        predictions = data['predictions']
        timestamps = data['timestamps']
        
        # Show current location
        current_loc = data.get('location', {'lat': lat, 'lon': lon})
        st.info(f"Showing forecast for location: {current_loc['lat']:.4f}, {current_loc['lon']:.4f}")
        
        # Create the main forecast plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predictions,
            mode='lines',
            name='Solar Energy Production',
            line=dict(color='#FFA500', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.1)'
        ))
        
        fig.update_layout(
            title="7-Day Solar Energy Production Forecast",
            xaxis_title="Date & Time",
            yaxis_title="Energy Production (kWh)",
            hovermode='x unified',
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Weekly Production", 
                f"{predictions.sum():.1f} kWh"
            )
        
        with col2:
            st.metric(
                "Average Daily Production", 
                f"{predictions.sum()/7:.1f} kWh"
            )
        
        with col3:
            st.metric(
                "Peak Production", 
                f"{predictions.max():.1f} kWh"
            )
        
        with col4:
            st.metric(
                "Best Production Day", 
                timestamps[predictions.argmax()].strftime("%A")
            )
        
        # Daily breakdown
        st.subheader("Daily Production Summary")
        
        # Group by day
        df_daily = pd.DataFrame({
            'timestamp': timestamps,
            'production': predictions
        })
        df_daily['date'] = df_daily['timestamp'].dt.date
        daily_summary = df_daily.groupby('date')['production'].sum().reset_index()
        daily_summary['day_name'] = pd.to_datetime(daily_summary['date']).dt.strftime('%A')
        
        # Display daily table
        st.dataframe(
            daily_summary[['date', 'day_name', 'production']].rename(columns={
                'date': 'Date',
                'day_name': 'Day',
                'production': 'Production (kWh)'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    else:
        st.info("Enter coordinates and the forecast will update automatically, or click 'Update Forecast'.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Weather data provided by Open-Meteo API*")

if __name__ == "__main__":
    main()