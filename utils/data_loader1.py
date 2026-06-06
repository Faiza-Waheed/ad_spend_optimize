import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle

@st.cache_data
def generate_dataset1():
    """Generate Marketing & Sales Revenue Dataset (60k rows)"""
    np.random.seed(42)
    n_samples = 100
    
    # Features
    ad_spend_online = np.random.uniform(100, 10000, n_samples)
    ad_spend_offline = np.random.uniform(50, 5000, n_samples)
    marketing_budget = ad_spend_online + ad_spend_offline + np.random.uniform(100, 2000, n_samples)
    discount_percentage = np.random.uniform(0, 50, n_samples)
    num_promotions = np.random.randint(0, 10, n_samples)
    
    # Categorical features
    regions = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    channels = np.random.choice(['Online', 'Offline', 'Both'], n_samples)
    categories = np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], n_samples)
    segments = np.random.choice(['Premium', 'Regular', 'Budget'], n_samples)
    seasons = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    
    # Target: sales_revenue with realistic relationships
    sales_revenue = (
        0.5 * ad_spend_online +
        0.3 * ad_spend_offline +
        50 * (discount_percentage / 10) +
        100 * num_promotions +
        np.where(regions == 'North', 500, 
                np.where(regions == 'South', 300, 
                        np.where(regions == 'East', 400, 350))) +
        np.where(channels == 'Both', 800, 
                np.where(channels == 'Online', 600, 400)) +
        np.where(segments == 'Premium', 1000,
                np.where(segments == 'Regular', 500, 200)) +
        np.random.normal(0, 500, n_samples)
    )
    
    # Add some missing values (3%)
    # Convert to list to modify
    ad_spend_online_list = ad_spend_online.tolist()
    discount_list = discount_percentage.tolist()
    promotions_list = num_promotions.tolist()
    
    for col_name, col_list in [('ad_spend_online', ad_spend_online_list), 
                                ('discount_percentage', discount_list), 
                                ('num_promotions', promotions_list)]:
        missing_idx = np.random.choice(n_samples, size=int(n_samples*0.03), replace=False)
        for idx in missing_idx:
            col_list[idx] = np.nan
    
    df = pd.DataFrame({
        'ad_spend_online_usd': ad_spend_online_list,
        'ad_spend_offline_usd': ad_spend_offline,
        'marketing_budget_usd': marketing_budget,
        'discount_percentage': discount_list,
        'num_promotions': promotions_list,
        'region': regions,
        'sales_channel': channels,
        'product_category': categories,
        'customer_segment': segments,
        'season': seasons,
        'sales_revenue_usd': sales_revenue
    })
    
    return df

@st.cache_data
def generate_dataset2():
    """Generate simpler Advertising dataset (TV/Radio/Newspaper + Sales)"""
    np.random.seed(42)
    n_samples = 8000
    
    tv = np.random.uniform(0, 300, n_samples)
    radio = np.random.uniform(0, 50, n_samples)
    newspaper = np.random.uniform(0, 100, n_samples)
    
    # Sales formula: TV has highest impact
    sales = (
        0.05 * tv +
        0.2 * radio +
        0.03 * newspaper +
        np.random.normal(0, 5, n_samples)
    ) * 100
    
    df = pd.DataFrame({
        'tv_ad_spend_usd': tv,
        'radio_ad_spend_usd': radio,
        'newspaper_ad_spend_usd': newspaper,
        'sales_revenue_usd': sales
    })
    
    return df

@st.cache_data
def load_and_prepare_data(dataset_choice, use_pretrained_split=True):
    """Load selected dataset and prepare train/test/future splits"""
    if dataset_choice == "Marketing_Analytics_Dataset_by_Slidescope":
        df = generate_dataset1()
        safe_name = 'marketing_and_sales_revenue_dataset_60k_rows'
    else:
        df = generate_dataset2()
        safe_name = 'simplified_advertising_dataset_20k_rows'
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Separate features and target
    target = 'sales_revenue_usd'
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Try to load pre-split data if available
    if use_pretrained_split:
        try:
            with open(f'saved_models/{safe_name}_test_data.pkl', 'rb') as f:
                test_data = pickle.load(f)
            
            # Use the same split as training
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            X_future = test_data['X_future']
            y_future = test_data['y_future']
            
            # For training data, we need to get the remaining indices
            # This is a bit tricky - simpler approach: split consistently
            # Let's just re-split with same random_state
            X_temp, X_future, y_temp, y_future = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42
            )
            
        except FileNotFoundError:
            # If pre-split data not found, create new split
            print(f"Pre-split data not found for {safe_name}, creating new split...")
            X_temp, X_future, y_temp, y_future = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42
            )
    else:
        # Create fresh split
        X_temp, X_future, y_temp, y_future = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42
        )
    
    return df, X_train, X_test, y_train, y_test, X_future, y_future