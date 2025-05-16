"""
Configuration file for Istanbul Traffic Analysis Project
Contains all project settings and parameters
"""

import os
from datetime import datetime

# Project Information
PROJECT_NAME = "Istanbul Traffic Analysis"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Data Science Student"
PROJECT_DESCRIPTION = "Analyzing traffic patterns in Istanbul using machine learning and statistical methods"

# Data Collection Settings
DATA_COLLECTION = {
    'start_date': '2023-01-01',
    'end_date': '2023-01-31',
    'location': 'Istanbul',
    'districts': ['Besiktas', 'Kadikoy', 'Sisli', 'Uskudar', 'Beyoglu'],
    'api_rate_limit': 0.1,  # seconds between API calls
    'weather_api_key': os.getenv('WEATHER_API_KEY', 'your_api_key_here')
}

# File Paths
PATHS = {
    'raw_traffic': 'data/raw/traffic_data.csv',
    'raw_weather': 'data/raw/weather_data.csv',
    'merged_data': 'data/processed/merged_data.csv',
    'enriched_data': 'data/processed/enriched_data.csv',
    'figures_dir': 'output/figures/',
    'reports_dir': 'output/reports/',
    'logs_dir': 'logs/'
}

# Data Processing Settings
PROCESSING = {
    'outlier_threshold': 3,  # standard deviations for outlier removal
    'missing_value_strategy': 'median',  # 'mean', 'median', 'mode'
    'datetime_format': '%Y-%m-%d %H:%M:%S',
    'categorical_encoding': 'label',  # 'label' or 'onehot'
}

# Feature Engineering
FEATURES = {
    'numerical_features': [
        'traffic_density', 'average_speed', 'vehicle_count',
        'temperature', 'humidity', 'precipitation', 'wind_speed',
        'hour', 'day_of_week', 'month', 'traffic_efficiency'
    ],
    'categorical_features': [
        'district', 'weather_category', 'time_period', 
        'temp_category', 'precipitation_category'
    ],
    'target_variables': ['traffic_density', 'weather_category'],
    'rush_hours': {
        'morning': (7, 9),
        'evening': (17, 19),
        'night': (22, 6)
    },
    'temperature_bins': [-float('inf'), 10, 20, 30, float('inf')],
    'precipitation_bins': [-float('inf'), 0, 5, 15, float('inf')]
}

# Statistical Analysis Settings
STATISTICS = {
    'significance_level': 0.05,
    'hypothesis_tests': [
        'weekday_vs_weekend_density',
        'weather_effect_on_speed',
        'precipitation_density_correlation',
        'rush_hour_congestion',
        'district_variations'
    ],
    'correlation_threshold': 0.3,  # minimum correlation to report
    'normality_test': 'shapiro',  # 'shapiro' or 'kolmogorov'
}

# Machine Learning Settings
ML_SETTINGS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    
    # Clustering
    'clustering': {
        'kmeans': {
            'max_clusters': 10,
            'n_init': 10
        },
        'dbscan': {
            'eps': 0.5,
            'min_samples': 5
        }
    },
    
    # PCA
    'pca': {
        'n_components': 5,
        'explained_variance_threshold': 0.95
    },
    
    # Models
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
    }
}

# Visualization Settings
VISUALIZATION = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'viridis',
    'save_format': 'png',
    'font_size': 12,
    'title_size': 16
}

# Reporting Settings
REPORTING = {
    'include_code': False,
    'include_data_summary': True,
    'include_visualizations': True,
    'output_format': 'html',  # 'html', 'pdf', 'markdown'
    'template': 'default'
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
    'log_file': 'logs/traffic_analysis.log'
}

# API URLs and Endpoints
API_ENDPOINTS = {
    'ibb_traffic': 'https://data.ibb.gov.tr/api/3/action/datastore_search',
    'weather_api': 'http://api.weatherapi.com/v1',
    'backup_weather': 'https://api.openweathermap.org/data/2.5'
}

# Data Quality Checks
QUALITY_CHECKS = {
    'min_records': 100,
    'max_missing_percentage': 0.1,  # 10% maximum missing values
    'required_columns': {
        'traffic': ['datetime', 'district', 'traffic_density', 'average_speed'],
        'weather': ['datetime', 'temperature', 'humidity', 'precipitation']
    },
    'data_types': {
        'datetime': 'datetime64',
        'traffic_density': 'float64',
        'average_speed': 'float64',
        'temperature': 'float64'
    }
}

# Environment-specific settings
ENVIRONMENT = {
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'use_sample_data': os.getenv('USE_SAMPLE_DATA', 'False').lower() == 'true',
    'parallel_processing': os.getenv('PARALLEL_PROCESSING', 'True').lower() == 'true',
    'memory_limit': os.getenv('MEMORY_LIMIT', '8GB'),
    'temp_dir': os.getenv('TEMP_DIR', '/tmp')
}

# Export settings for easy access
def get_config():
    """Return complete configuration dictionary"""
    return {
        'project': {
            'name': PROJECT_NAME,
            'version': PROJECT_VERSION,
            'author': PROJECT_AUTHOR,
            'description': PROJECT_DESCRIPTION
        },
        'data_collection': DATA_COLLECTION,
        'paths': PATHS,
        'processing': PROCESSING,
        'features': FEATURES,
        'statistics': STATISTICS,
        'ml_settings': ML_SETTINGS,
        'visualization': VISUALIZATION,
        'reporting': REPORTING,
        'logging': LOGGING,
        'api_endpoints': API_ENDPOINTS,
        'quality_checks': QUALITY_CHECKS,
        'environment': ENVIRONMENT
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    
    print("=" * 60)
    print("ISTANBUL TRAFFIC ANALYSIS - CONFIGURATION")
    print("=" * 60)
    
    for section, settings in config.items():
        print(f"\n{section.upper().replace('_', ' ')}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print_config()