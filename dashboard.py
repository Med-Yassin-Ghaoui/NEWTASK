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
        # Load model
        with open('solar_energy_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # Load feature names
        with open('features_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}. Please ensure 'solar_energy_model.pkl' and 'feature_names.pkl' are in the same directory.")
        return None, None
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
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "shortwave_radiation",
            
        ],
        "forecast_days": 7,
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Weather API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error fetching weather data: {e}")
        return None

def process_weather_data(weather_data, feature_names):
    """Convert weather API response to model input format with proper feature mapping"""
    if not weather_data or 'hourly' not in weather_data:
        return None, None
        
    try:
        hourly = weather_data['hourly']
        
        # Create DataFrame from weather data
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temperature_2m': hourly.get('temperature_2m', [0] * len(hourly['time'])),
            'precipitation': hourly.get('precipitation', [0] * len(hourly['time'])),
            'cloud_cover': hourly.get('cloud_cover', [0] * len(hourly['time'])),
            'wind_speed_10m': hourly.get('wind_speed_10m', [0] * len(hourly['time'])),
            'shortwave_radiation': hourly.get('shortwave_radiation', [0] * len(hourly['time'])),
            
        })
        
        # Add time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.dayofyear  # Day of year (1-365/366) - this matches your model
        df['month'] = df['datetime'].dt.month
        
        # No timestamp feature needed for this model
        
        # H_sun will be handled in the feature mapping loop if it exists in the model
        
        # Initialize feature array with zeros
        X = np.zeros((len(df), len(feature_names)))
        
        # Create exact mapping based on your actual model's features
        # Your model features: ['Basel Precipitation Total', 'Basel Cloud Cover Total', 
        # 'Basel Shortwave Radiation', 'T2m', 'WS10m', 'hour', 'day', 'month']
        
        for i, feature in enumerate(feature_names):
            if feature == 'Basel Precipitation Total':
                X[:, i] = df['precipitation'].values
            elif feature == 'Basel Cloud Cover Total':
                X[:, i] = df['cloud_cover'].values
            elif feature == 'Basel Shortwave Radiation':
                X[:, i] = df['shortwave_radiation'].values
            elif feature == 'T2m':
                X[:, i] = df['temperature_2m'].values
            elif feature == 'WS10m':
                X[:, i] = df['wind_speed_10m'].values
            elif feature == 'hour':
                X[:, i] = df['hour'].values
            elif feature == 'day':
                X[:, i] = df['day'].values  # Day of year (1-365/366)
            elif feature == 'month':
                X[:, i] = df['month'].values
            else:
                # Handle any unexpected features with reasonable defaults
                st.warning(f"Unknown feature encountered: {feature}. Using default value 0.")
                X[:, i] = 0
        
        # Data quality checks and corrections
        # Replace any NaN values with appropriate defaults
        for i, feature in enumerate(feature_names):
            col_data = X[:, i]
            if np.any(np.isnan(col_data)) or np.any(np.isinf(col_data)):
                if 'radiation' in feature.lower():
                    X[:, i] = np.nan_to_num(col_data, nan=0.0, posinf=1000.0, neginf=0.0)
                elif 'precipitation' in feature.lower():
                    X[:, i] = np.nan_to_num(col_data, nan=0.0, posinf=100.0, neginf=0.0)
                elif 'temperature' in feature.lower() or 'T2m' in feature:
                    X[:, i] = np.nan_to_num(col_data, nan=20.0, posinf=50.0, neginf=-20.0)
                elif 'wind' in feature.lower() or 'WS' in feature:
                    X[:, i] = np.nan_to_num(col_data, nan=5.0, posinf=30.0, neginf=0.0)
                elif 'cloud' in feature.lower():
                    X[:, i] = np.nan_to_num(col_data, nan=50.0, posinf=100.0, neginf=0.0)
                else:
                    X[:, i] = np.nan_to_num(col_data, nan=0.0)
        
        return X, df['datetime']
    
    except Exception as e:
        st.error(f"Error processing weather data: {e}")
        return None, None

def main():
    # Header
    st.title("☀️ Solar Energy Production Forecast")
    st.markdown("7-day solar energy production prediction based on weather forecast")
    
    # Sidebar for location input
    st.sidebar.header("Location Settings")
    lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f", min_value=-90.0, max_value=90.0)
    lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f", min_value=-180.0, max_value=180.0)
    
    # Display current feature names and mapping for debugging
    if st.sidebar.checkbox("Show Feature Mapping"):
        try:
            _, feature_names = load_model_and_features()
            if feature_names:
                st.sidebar.write("**Model Features → API Data Mapping:**")
                feature_mapping_display = {
                    'Basel Precipitation Total': 'precipitation (mm)',
                    'Basel Cloud Cover Total': 'cloud_cover (%)',
                    'Basel Shortwave Radiation': 'shortwave_radiation (W/m²)',
                    'T2m': 'temperature_2m (°C)',
                    'WS10m': 'wind_speed_10m (m/s)',
                    'hour': 'hour of day (0-23)',
                    'day': 'day of year (1-365/366)',
                    'month': 'month (1-12)'
                }
                
                for feature in feature_names:
                    mapping = feature_mapping_display.get(feature, 'Unknown mapping')
                    st.sidebar.write(f"• **{feature}** ← {mapping}")
        except:
            st.sidebar.write("Could not load feature mapping")
    
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
            
            if model is None or feature_names is None:
                st.error("Could not load the solar energy model. Please check your model files.")
                st.info("Expected files: 'solar_energy_model.pkl' and 'feature_names.pkl'")
                return
            
            # Display feature count info
            st.info(f"Model expects {len(feature_names)} features: {', '.join(feature_names)}")
            
            # Get weather forecast
            weather_data = get_weather_forecast(lat, lon)
            
            if weather_data is None:
                st.error("Could not fetch weather data. Please try again.")
                return
            
            # Process data for prediction
            X, timestamps = process_weather_data(weather_data, feature_names)
            
            if X is None or timestamps is None:
                st.error("Could not process weather data.")
                return
            
            # Display data shape for debugging
            st.info(f"Processed data shape: {X.shape} (should be [n_samples, {len(feature_names)}])")
            
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
                    'processed_features': X,
                    'location': {'lat': lat, 'lon': lon}
                }
                
                # Show success message when location changes
                if location_changed and not update_clicked:
                    st.success(f"Forecast updated for new location: {lat:.4f}, {lon:.4f}")
                elif update_clicked:
                    st.success("Forecast updated successfully!")
                
            except Exception as e:
                st.error(f"Error making predictions: {e}")
                st.info("This might be due to feature mismatch between the model and processed data.")
                st.info(f"Expected features: {len(feature_names)}, Got: {X.shape[1] if X is not None else 'None'}")
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
            max_idx = predictions.argmax()
            if max_idx < len(timestamps):
                best_day = timestamps.iloc[max_idx].strftime("%A") if hasattr(timestamps, 'iloc') else timestamps[max_idx].strftime("%A")
            else:
                best_day = "N/A"
            st.metric(
                "Best Production Day", 
                best_day
            )
        
        # Daily breakdown
        st.subheader("Daily Production Summary")
        
       
        df_daily = pd.DataFrame({
            'timestamp': timestamps,
            'production': predictions
        })
        df_daily['date'] = pd.to_datetime(df_daily['timestamp']).dt.date
        daily_summary = df_daily.groupby('date')['production'].sum().reset_index()
        daily_summary['day_name'] = pd.to_datetime(daily_summary['date']).dt.strftime('%A')
        daily_summary['production'] = daily_summary['production'].round(2)
        
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
        
        # Feature values preview (for debugging)
        if st.checkbox("Show Processed Feature Values (First 24 Hours)"):
            feature_df = pd.DataFrame(data['processed_features'][:24], columns=data.get('feature_names', [f'Feature_{i}' for i in range(data['processed_features'].shape[1])]))
            feature_df.insert(0, 'Time', timestamps[:24])
            st.dataframe(feature_df, use_container_width=True)
        
        # Weather data preview (optional)
        if st.checkbox("Show Weather Data Details"):
            weather_df = pd.DataFrame({
                'Time': timestamps,
                'Temperature (°C)': data['weather_data']['hourly']['temperature_2m'],
                'Cloud Cover (%)': data['weather_data']['hourly']['cloud_cover'],
                'Precipitation (mm)': data['weather_data']['hourly']['precipitation'],
                'Wind Speed (m/s)': data['weather_data']['hourly']['wind_speed_10m'],
                'Shortwave Radiation (W/m²)': data['weather_data']['hourly']['shortwave_radiation']
            })
            st.dataframe(weather_df.head(24), use_container_width=True)  # Show first 24 hours
    
    else:
        st.info("Enter coordinates and the forecast will update automatically, or click 'Update Forecast'.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Weather data provided by Open-Meteo API*")

if __name__ == "__main__":
    main()