import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE (%)': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred

def save_model(model, filename):
    """Save model to file"""
    os.makedirs('saved_models', exist_ok=True)
    with open(f'saved_models/{filename}', 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load model from file"""
    with open(f'saved_models/{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

def load_pretrained_models(dataset_choice):
    """Load pre-trained models for the selected dataset"""
    # Create safe filename
    safe_name = dataset_choice.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')
    model_path = f'saved_models/{safe_name}_models.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    else:
        # If models don't exist, return None and they'll be trained on the fly
        return None

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import pickle
# import os

# def train_linear_regression(X_train, y_train):
#     """Train Linear Regression model"""
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     return model

# def train_xgboost(X_train, y_train):
#     """Train XGBoost model"""
#     model = XGBRegressor(
#         n_estimators=100,
#         learning_rate=0.1,
#         max_depth=5,
#         random_state=42,
#         verbosity=0
#     )
#     model.fit(X_train, y_train)
#     return model

# def train_random_forest(X_train, y_train):
#     """Train Random Forest model"""
#     model = RandomForestRegressor(
#         n_estimators=100,
#         max_depth=10,
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test):
#     """Evaluate model performance"""
#     y_pred = model.predict(X_test)
    
#     metrics = {
#         'R² Score': r2_score(y_test, y_pred),
#         'MSE': mean_squared_error(y_test, y_pred),
#         'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'MAE': mean_absolute_error(y_test, y_pred),
#         'MAPE (%)': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
#     }
    
#     return metrics, y_pred

# def save_model(model, filename):
#     """Save model to file"""
#     os.makedirs('saved_models', exist_ok=True)
#     with open(f'saved_models/{filename}', 'wb') as f:
#         pickle.dump(model, f)

# def load_model(filename):
#     """Load model from file"""
#     with open(f'saved_models/{filename}', 'rb') as f:
#         model = pickle.load(f)
#     return model