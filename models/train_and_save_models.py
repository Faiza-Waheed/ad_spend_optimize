import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from utils.data_loader import load_and_prepare_data
from models.train_model import *

def train_and_save_models():
    """Train models for both datasets and save them"""
    
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    datasets = [
        "Marketing & Sales Revenue Dataset (60k rows)",
        "Simplified Advertising Dataset (20k rows)"
    ]
    
    for dataset_name in datasets:
        print(f"\n📊 Training models for: {dataset_name}")
        
        # Load and prepare data
        df, X_train, X_test, y_train, y_test, X_future, y_future = load_and_prepare_data(dataset_name)
        
        # Train models
        print("  - Training Linear Regression...")
        lr_model = train_linear_regression(X_train, y_train)
        
        print("  - Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        
        print("  - Training XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        
        # Evaluate and print metrics
        lr_metrics, _ = evaluate_model(lr_model, X_test, y_test)
        rf_metrics, _ = evaluate_model(rf_model, X_test, y_test)
        xgb_metrics, _ = evaluate_model(xgb_model, X_test, y_test)
        
        print(f"\n  📈 Performance on {dataset_name}:")
        print(f"    Linear Regression - R²: {lr_metrics['R² Score']:.4f}")
        print(f"    Random Forest     - R²: {rf_metrics['R² Score']:.4f}")
        print(f"    XGBoost           - R²: {xgb_metrics['R² Score']:.4f}")
        
        # Create a safe filename from dataset name
        safe_name = dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')
        
        # Save models
        model_data = {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'feature_names': X_train.columns.tolist(),
            'metrics': {
                'linear_regression': lr_metrics,
                'random_forest': rf_metrics,
                'xgboost': xgb_metrics
            }
        }
        
        with open(f'saved_models/{safe_name}_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"  ✅ Models saved to saved_models/{safe_name}_models.pkl")
        
        # Also save the test/future data for consistent evaluation
        test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'X_future': X_future,
            'y_future': y_future
        }
        
        with open(f'saved_models/{safe_name}_test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"  ✅ Test data saved to saved_models/{safe_name}_test_data.pkl")

if __name__ == "__main__":
    print("🚀 Starting offline model training...")
    train_and_save_models()
    print("\n✅ All models trained and saved successfully!")