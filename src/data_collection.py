"""
Istanbul Traffic Data Collection Module

This module handles data collection from various APIs including:
- Istanbul Metropolitan Municipality traffic data
- Weather data from WeatherAPI.com
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

class TrafficDataCollector:
    """Collect traffic data from Istanbul Metropolitan Municipality API"""
    
    def __init__(self):
        self.base_url = "https://data.ibb.gov.tr/api"
        self.headers = {'User-Agent': 'Istanbul-Traffic-Analysis'}
        
    def get_traffic_data(self, start_date, end_date):
        """
        Collect traffic data for specified date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Traffic data
        """
        traffic_data = []
        
        # Simulated data collection logic
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_datetime:
            # Simulate API call for each day
            daily_data = self._get_daily_traffic(current_date)
            traffic_data.extend(daily_data)
            current_date += timedelta(days=1)
            time.sleep(0.1)  # Rate limiting
            
        return pd.DataFrame(traffic_data)
    
    def _get_daily_traffic(self, date):
        """Get traffic data for a specific day"""
        # This would be replaced with actual API calls
        daily_data = []
        for hour in range(24):
            for district in ['Besiktas', 'Kadikoy', 'Sisli', 'Uskudar', 'Beyoglu']:
                daily_data.append({
                    'datetime': date.replace(hour=hour),
                    'district': district,
                    'traffic_density': 50 + (hour % 8) * 10,  # Simulated data
                    'average_speed': 30 + (hour % 6) * 5,
                    'vehicle_count': 100 + (hour % 10) * 20
                })
        return daily_data

class WeatherDataCollector:
    """Collect weather data from WeatherAPI.com"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "http://api.weatherapi.com/v1"
        
    def get_weather_data(self, start_date, end_date, location="Istanbul"):
        """
        Collect weather data for specified date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            location (str): Location name
            
        Returns:
            pd.DataFrame: Weather data
        """
        weather_data = []
        
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_datetime:
            daily_weather = self._get_daily_weather(current_date, location)
            weather_data.extend(daily_weather)
            current_date += timedelta(days=1)
            time.sleep(0.2)  # Rate limiting
            
        return pd.DataFrame(weather_data)
    
    def _get_daily_weather(self, date, location):
        """Get weather data for a specific day"""
        # Simulated weather data
        daily_data = []
        for hour in range(24):
            daily_data.append({
                'datetime': date.replace(hour=hour),
                'temperature': 15 + (hour % 12) * 2,
                'humidity': 60 + (hour % 8) * 5,
                'precipitation': max(0, (hour % 6) - 3),
                'wind_speed': 10 + (hour % 4) * 3,
                'weather_condition': 'Clear' if hour % 8 < 6 else 'Rainy'
            })
        return daily_data

def main():
    """Main data collection pipeline"""
    print("Starting Istanbul Traffic Data Collection...")
    
    # Initialize collectors
    traffic_collector = TrafficDataCollector()
    weather_collector = WeatherDataCollector()
    
    # Define date range
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Collect data
    print("Collecting traffic data...")
    traffic_df = traffic_collector.get_traffic_data(start_date, end_date)
    
    print("Collecting weather data...")
    weather_df = weather_collector.get_weather_data(start_date, end_date)
    
    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    traffic_df.to_csv('data/raw/traffic_data.csv', index=False)
    weather_df.to_csv('data/raw/weather_data.csv', index=False)
    
    print(f"Data collection completed!")
    print(f"Traffic records: {len(traffic_df)}")
    print(f"Weather records: {len(weather_df)}")

if __name__ == "__main__":
    main() 