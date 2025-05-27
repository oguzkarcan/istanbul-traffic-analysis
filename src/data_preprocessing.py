"""
Data Preprocessing Module for Istanbul Traffic Analysis

This module handles:
- Data cleaning and validation
- Missing value treatment
- Feature engineering
- Data transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handle data preprocessing for traffic and weather data"""
    
    def __init__(self):
        self.traffic_data = None
        self.weather_data = None
        self.merged_data = None
        self.enriched_data = None
        
    def load_data(self, traffic_path='data/raw/traffic_data.csv', 
                  weather_path='data/raw/weather_data.csv'):
        """Load raw data from CSV files"""
        print("Loading raw data...")
        
        self.traffic_data = pd.read_csv(traffic_path)
        self.weather_data = pd.read_csv(weather_path)
        
        # Convert datetime columns
        self.traffic_data['datetime'] = pd.to_datetime(self.traffic_data['datetime'])
        self.weather_data['datetime'] = pd.to_datetime(self.weather_data['datetime'])
        
        print(f"Traffic data shape: {self.traffic_data.shape}")
        print(f"Weather data shape: {self.weather_data.shape}")
        
    def clean_traffic_data(self):
        """Clean and validate traffic data"""
        print("Cleaning traffic data...")
        
        # Remove duplicates
        initial_count = len(self.traffic_data)
        self.traffic_data = self.traffic_data.drop_duplicates()
        print(f"Removed {initial_count - len(self.traffic_data)} duplicate records")
        
        # Handle missing values
        self.traffic_data['traffic_density'].fillna(
            self.traffic_data['traffic_density'].median(), inplace=True)
        self.traffic_data['average_speed'].fillna(
            self.traffic_data['average_speed'].median(), inplace=True)
        self.traffic_data['vehicle_count'].fillna(
            self.traffic_data['vehicle_count'].median(), inplace=True)
        
        # Remove outliers (values beyond 3 standard deviations)
        for col in ['traffic_density', 'average_speed', 'vehicle_count']:
            mean = self.traffic_data[col].mean()
            std = self.traffic_data[col].std()
            self.traffic_data = self.traffic_data[
                (self.traffic_data[col] >= mean - 3*std) & 
                (self.traffic_data[col] <= mean + 3*std)
            ]
            
    def clean_weather_data(self):
        """Clean and validate weather data"""
        print("Cleaning weather data...")
        
        # Remove duplicates
        initial_count = len(self.weather_data)
        self.weather_data = self.weather_data.drop_duplicates()
        print(f"Removed {initial_count - len(self.weather_data)} duplicate records")
        
        # Handle missing values
        self.weather_data['temperature'].fillna(
            self.weather_data['temperature'].median(), inplace=True)
        self.weather_data['humidity'].fillna(
            self.weather_data['humidity'].median(), inplace=True)
        self.weather_data['precipitation'].fillna(0, inplace=True)
        self.weather_data['wind_speed'].fillna(
            self.weather_data['wind_speed'].median(), inplace=True)
        
        # Validate ranges
        self.weather_data['humidity'] = self.weather_data['humidity'].clip(0, 100)
        self.weather_data['precipitation'] = self.weather_data['precipitation'].clip(0, None)
        
    def merge_datasets(self):
        """Merge traffic and weather data on datetime"""
        print("Merging traffic and weather data...")
        
        # Round datetime to nearest hour for better matching
        self.traffic_data['datetime_hour'] = self.traffic_data['datetime'].dt.round('H')
        self.weather_data['datetime_hour'] = self.weather_data['datetime'].dt.round('H')
        
        # Merge on rounded datetime
        self.merged_data = pd.merge(
            self.traffic_data,
            self.weather_data,
            on='datetime_hour',
            how='inner',
            suffixes=('_traffic', '_weather')
        )
        
        # Use traffic datetime as primary
        self.merged_data['datetime'] = self.merged_data['datetime_traffic']
        self.merged_data = self.merged_data.drop(['datetime_traffic', 'datetime_weather', 'datetime_hour'], axis=1)
        
        print(f"Merged data shape: {self.merged_data.shape}")
        
    def engineer_features(self):
        """Create additional features for analysis"""
        print("Engineering features...")
        
        self.enriched_data = self.merged_data.copy()
        
        # Temporal features
        self.enriched_data['hour'] = self.enriched_data['datetime'].dt.hour
        self.enriched_data['day_of_week'] = self.enriched_data['datetime'].dt.dayofweek
        self.enriched_data['month'] = self.enriched_data['datetime'].dt.month
        self.enriched_data['is_weekend'] = self.enriched_data['day_of_week'].isin([5, 6])
        
        # Rush hour classification
        def classify_rush_hour(hour):
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 'Rush'
            elif 22 <= hour <= 6:
                return 'Night'
            else:
                return 'Normal'
                
        self.enriched_data['time_period'] = self.enriched_data['hour'].apply(classify_rush_hour)
        
        # Weather categorization
        def categorize_weather(condition):
            if pd.isna(condition):
                return 'Unknown'
            condition = condition.lower()
            if 'rain' in condition or 'storm' in condition:
                return 'Rainy'
            elif 'cloud' in condition:
                return 'Cloudy'
            elif 'clear' in condition or 'sunny' in condition:
                return 'Clear'
            else:
                return 'Other'
                
        self.enriched_data['weather_category'] = self.enriched_data['weather_condition'].apply(categorize_weather)
        
        # Temperature categories
        self.enriched_data['temp_category'] = pd.cut(
            self.enriched_data['temperature'],
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=['Cold', 'Cool', 'Warm', 'Hot']
        )
        
        # Traffic efficiency metric
        self.enriched_data['traffic_efficiency'] = (
            self.enriched_data['average_speed'] / 
            (self.enriched_data['traffic_density'] + 1)
        )
        
        # Precipitation categories
        self.enriched_data['precipitation_category'] = pd.cut(
            self.enriched_data['precipitation'],
            bins=[-np.inf, 0, 5, 15, np.inf],
            labels=['None', 'Light', 'Moderate', 'Heavy']
        )
        
        print(f"Enriched data shape: {self.enriched_data.shape}")
        print(f"New features created: {len(self.enriched_data.columns) - len(self.merged_data.columns)}")
        
    def save_processed_data(self):
        """Save cleaned and processed data"""
        print("Saving processed data...")
        
        # Save merged data
        self.merged_data.to_csv('data/processed/merged_data.csv', index=False)
        print("Merged data saved to data/processed/merged_data.csv")
        
        # Save enriched data
        self.enriched_data.to_csv('data/processed/enriched_data.csv', index=False)
        print("Enriched data saved to data/processed/enriched_data.csv")
        
    def get_data_summary(self):
        """Generate summary statistics"""
        print("\n=== DATA SUMMARY ===")
        
        if self.enriched_data is not None:
            print(f"Final dataset shape: {self.enriched_data.shape}")
            print(f"Date range: {self.enriched_data['datetime'].min()} to {self.enriched_data['datetime'].max()}")
            print(f"Districts covered: {self.enriched_data['district'].nunique()}")
            print(f"Missing values: {self.enriched_data.isnull().sum().sum()}")
            
            print("\nTraffic metrics summary:")
            traffic_cols = ['traffic_density', 'average_speed', 'vehicle_count']
            print(self.enriched_data[traffic_cols].describe())
            
            print("\nWeather metrics summary:")
            weather_cols = ['temperature', 'humidity', 'precipitation', 'wind_speed']
            print(self.enriched_data[weather_cols].describe())

def main():
    """Main preprocessing pipeline"""
    print("Starting data preprocessing pipeline...\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    preprocessor.load_data()
    
    # Clean data
    preprocessor.clean_traffic_data()
    preprocessor.clean_weather_data()
    
    # Merge datasets
    preprocessor.merge_datasets()
    
    # Engineer features
    preprocessor.engineer_features()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Generate summary
    preprocessor.get_data_summary()
    
    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main() 