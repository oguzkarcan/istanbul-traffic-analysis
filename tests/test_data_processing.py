"""
Unit tests for data processing module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        """Set up test data"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample traffic data
        self.sample_traffic = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=24, freq='H'),
            'district': ['Besiktas'] * 24,
            'traffic_density': np.random.normal(50, 10, 24),
            'average_speed': np.random.normal(30, 5, 24),
            'vehicle_count': np.random.normal(100, 20, 24)
        })
        
        # Create sample weather data
        self.sample_weather = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=24, freq='H'),
            'temperature': np.random.normal(20, 5, 24),
            'humidity': np.random.normal(60, 10, 24),
            'precipitation': np.random.exponential(2, 24),
            'wind_speed': np.random.normal(15, 3, 24),
            'weather_condition': ['Clear'] * 12 + ['Rainy'] * 12
        })
        
        # Set test data
        self.preprocessor.traffic_data = self.sample_traffic.copy()
        self.preprocessor.weather_data = self.sample_weather.copy()
    
    def test_clean_traffic_data(self):
        """Test traffic data cleaning"""
        # Add some missing values
        self.preprocessor.traffic_data.loc[0, 'traffic_density'] = np.nan
        self.preprocessor.traffic_data.loc[1, 'average_speed'] = np.nan
        
        initial_shape = self.preprocessor.traffic_data.shape
        self.preprocessor.clean_traffic_data()
        
        # Check that missing values are handled
        self.assertFalse(self.preprocessor.traffic_data['traffic_density'].isnull().any())
        self.assertFalse(self.preprocessor.traffic_data['average_speed'].isnull().any())
        
        # Check that shape is maintained or reduced (due to outlier removal)
        self.assertLessEqual(self.preprocessor.traffic_data.shape[0], initial_shape[0])
    
    def test_clean_weather_data(self):
        """Test weather data cleaning"""
        # Add some missing values
        self.preprocessor.weather_data.loc[0, 'temperature'] = np.nan
        self.preprocessor.weather_data.loc[1, 'humidity'] = np.nan
        
        self.preprocessor.clean_weather_data()
        
        # Check that missing values are handled
        self.assertFalse(self.preprocessor.weather_data['temperature'].isnull().any())
        self.assertFalse(self.preprocessor.weather_data['humidity'].isnull().any())
        
        # Check humidity range validation
        self.assertTrue((self.preprocessor.weather_data['humidity'] >= 0).all())
        self.assertTrue((self.preprocessor.weather_data['humidity'] <= 100).all())
    
    def test_merge_datasets(self):
        """Test dataset merging"""
        self.preprocessor.merge_datasets()
        
        # Check that merged data exists
        self.assertIsNotNone(self.preprocessor.merged_data)
        
        # Check that merged data has both traffic and weather columns
        traffic_cols = ['traffic_density', 'average_speed', 'vehicle_count']
        weather_cols = ['temperature', 'humidity', 'precipitation']
        
        for col in traffic_cols + weather_cols:
            self.assertIn(col, self.preprocessor.merged_data.columns)
    
    def test_engineer_features(self):
        """Test feature engineering"""
        self.preprocessor.merge_datasets()
        self.preprocessor.engineer_features()
        
        # Check that new features are created
        expected_features = ['hour', 'day_of_week', 'month', 'is_weekend', 
                           'time_period', 'weather_category', 'temp_category', 
                           'traffic_efficiency', 'precipitation_category']
        
        for feature in expected_features:
            self.assertIn(feature, self.preprocessor.enriched_data.columns)
        
        # Check that hour feature is correct
        self.assertTrue((self.preprocessor.enriched_data['hour'] >= 0).all())
        self.assertTrue((self.preprocessor.enriched_data['hour'] <= 23).all())
        
        # Check that day_of_week is correct
        self.assertTrue((self.preprocessor.enriched_data['day_of_week'] >= 0).all())
        self.assertTrue((self.preprocessor.enriched_data['day_of_week'] <= 6).all())

class TestFeatureEngineering(unittest.TestCase):
    """Test specific feature engineering functions"""
    
    def test_rush_hour_classification(self):
        """Test rush hour classification logic"""
        def classify_rush_hour(hour):
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 'Rush'
            elif 22 <= hour <= 6:
                return 'Night'
            else:
                return 'Normal'
        
        # Test specific hours
        self.assertEqual(classify_rush_hour(8), 'Rush')
        self.assertEqual(classify_rush_hour(18), 'Rush')
        self.assertEqual(classify_rush_hour(23), 'Night')
        self.assertEqual(classify_rush_hour(2), 'Night')
        self.assertEqual(classify_rush_hour(12), 'Normal')
    
    def test_weather_categorization(self):
        """Test weather categorization logic"""
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
        
        # Test categorization
        self.assertEqual(categorize_weather('Clear'), 'Clear')
        self.assertEqual(categorize_weather('Rain'), 'Rainy')
        self.assertEqual(categorize_weather('Cloudy'), 'Cloudy')
        self.assertEqual(categorize_weather('Fog'), 'Other')
        self.assertEqual(categorize_weather(np.nan), 'Unknown')

if __name__ == '__main__':
    unittest.main() 