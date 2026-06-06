import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_and_prepare_data
from utils.visualization import *
from models.train_model import evaluate_model
import plotly.express as px

# Get dataset from session state
dataset_choice = st.session_state.get('dataset_choice', "Marketing & Sales Revenue Dataset (60k rows)")

# Load data
@st.cache_data
def load_data(dataset_choice):
    df, X_train, X_test, y_train, y_test, X_future, y_future = load_and_prepare_data(dataset_choice)
    return df, X_train, X_test, y_train, y_test, X_future, y_future

df, X_train, X_test, y_train, y_test, X_future, y_future = load_data(dataset_choice)

# Load models from session state or train if needed
if st.session_state.get('pretrained_models'):
    model_data = st.session_state['pretrained_models']
    lr_model = model_data['linear_regression']
    rf_model = model_data['random_forest']
    xgb_model = model_data['xgboost']
    
    # Get metrics from saved data
    lr_metrics = model_data['metrics']['linear_regression']
    rf_metrics = model_data['metrics']['random_forest']
    xgb_metrics = model_data['metrics']['xgboost']
    
    # Get predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
else:
    # Fallback: train on the fly (should not happen if we pre-trained)
    from models.train_model import train_linear_regression, train_random_forest, train_xgboost, evaluate_model
    
    with st.spinner("Training models (first time only)..."):
        lr_model = train_linear_regression(X_train, y_train)
        rf_model = train_random_forest(X_train, y_train)
        xgb_model = train_xgboost(X_train, y_train)
        
        lr_metrics, lr_pred = evaluate_model(lr_model, X_test, y_test)
        rf_metrics, rf_pred = evaluate_model(rf_model, X_test, y_test)
        xgb_metrics, xgb_pred = evaluate_model(xgb_model, X_test, y_test)

st.header("🤖 Model Performance Comparison")

# Rest of the page remains the same...
# [Continue with the same content as before]