"""
Main execution script for Istanbul Traffic Analysis
Complete data science pipeline from data collection to analysis
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection import main as collect_data
from data_preprocessing import main as preprocess_data
from statistical_analysis import main as run_statistics
from ml_models import main as run_ml_analysis

def create_output_directories():
    """Create necessary output directories"""
    directories = [
        'data/raw',
        'data/processed',
        'output/figures',
        'output/reports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def log_pipeline_step(step_name, status="STARTED"):
    """Log pipeline execution steps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {step_name}: {status}"
    
    print(log_message)
    
    # Also log to file
    with open('logs/pipeline.log', 'a') as f:
        f.write(log_message + '\n')

def run_complete_pipeline():
    """Run the complete data science pipeline"""
    print("=" * 80)
    print("ISTANBUL TRAFFIC ANALYSIS - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Create directories
    create_output_directories()
    
    try:
        # Step 1: Data Collection
        log_pipeline_step("Data Collection")
        collect_data()
        log_pipeline_step("Data Collection", "COMPLETED")
        print("\n" + "="*50 + "\n")
        
        # Step 2: Data Preprocessing
        log_pipeline_step("Data Preprocessing")
        preprocess_data()
        log_pipeline_step("Data Preprocessing", "COMPLETED")
        print("\n" + "="*50 + "\n")
        
        # Step 3: Statistical Analysis
        log_pipeline_step("Statistical Analysis")
        run_statistics()
        log_pipeline_step("Statistical Analysis", "COMPLETED")
        print("\n" + "="*50 + "\n")
        
        # Step 4: Machine Learning Analysis
        log_pipeline_step("Machine Learning Analysis")
        run_ml_analysis()
        log_pipeline_step("Machine Learning Analysis", "COMPLETED")
        print("\n" + "="*50 + "\n")
        
        # Pipeline completed
        log_pipeline_step("Complete Pipeline", "COMPLETED")
        print("=" * 80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"Completed at: {datetime.now()}")
        print("=" * 80)
        
    except Exception as e:
        log_pipeline_step("Pipeline", f"FAILED: {str(e)}")
        print(f"Pipeline failed with error: {str(e)}")
        raise

def run_individual_step(step):
    """Run individual pipeline step"""
    create_output_directories()
    
    steps = {
        'collect': collect_data,
        'preprocess': preprocess_data,
        'statistics': run_statistics,
        'ml': run_ml_analysis
    }
    
    if step in steps:
        print(f"Running {step} step...")
        log_pipeline_step(f"Individual Step: {step}")
        steps[step]()
        log_pipeline_step(f"Individual Step: {step}", "COMPLETED")
    else:
        print(f"Unknown step: {step}")
        print(f"Available steps: {list(steps.keys())}")

def print_project_info():
    """Print project information"""
    print("=" * 80)
    print("ISTANBUL TRAFFIC ANALYSIS PROJECT")
    print("=" * 80)
    print()
    print("This project analyzes traffic patterns in Istanbul using:")
    print("• Traffic data from Istanbul Metropolitan Municipality")
    print("• Weather data from WeatherAPI.com")
    print()
    print("Analysis includes:")
    print("• Exploratory Data Analysis (EDA)")
    print("• Statistical hypothesis testing")
    print("• Machine learning clustering and prediction")
    print("• Principal Component Analysis (PCA)")
    print()
    print("Research Questions:")
    print("• How do weather conditions affect traffic patterns?")
    print("• Are there significant differences between weekday/weekend traffic?")
    print("• What are the main factors influencing traffic congestion?")
    print("• Can we predict traffic density based on various factors?")
    print()
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Istanbul Traffic Analysis Pipeline')
    parser.add_argument('--step', type=str, help='Run specific step: collect, preprocess, statistics, ml')
    parser.add_argument('--info', action='store_true', help='Show project information')
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    if args.info:
        print_project_info()
    elif args.step:
        run_individual_step(args.step)
    elif args.full or len(sys.argv) == 1:
        # Run full pipeline if no arguments or --full flag
        run_complete_pipeline()
    else:
        parser.print_help() 