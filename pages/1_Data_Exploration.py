import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_and_prepare_data
from utils.visualization import *
import plotly.express as px

# Get dataset from session state
dataset_choice = st.session_state.get('dataset_choice', "Marketing & Sales Revenue Dataset (60k rows)")

# Load data (cached)
@st.cache_resource
def load_data(dataset_choice):
    df, X_train, X_test, y_train, y_test, X_future, y_future = load_and_prepare_data(dataset_choice)
    return df, X_train, X_test, y_train, y_test, X_future, y_future

df, X_train, X_test, y_train, y_test, X_future, y_future = load_data(dataset_choice)

st.header("📊 Data Exploration & Visualization")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Quick Stats")
    st.write(f"💰 **Total Sales Revenue:** ${df['sales_revenue_usd'].sum():,.0f}")
    st.write(f"📊 **Avg Sales Revenue:** ${df['sales_revenue_usd'].mean():,.0f}")
    st.write(f"📈 **Max Sales:** ${df['sales_revenue_usd'].max():,.0f}")
    st.write(f"📉 **Min Sales:** ${df['sales_revenue_usd'].min():,.0f}")

st.markdown("---")

# Visualizations
viz_option = st.selectbox(
    "Select Visualization",
    ["Correlation Heatmap", "Ad Spend vs Revenue Analysis", 
     "Sales Distribution", "Feature Distributions", "Scatter Plot Analysis"]
)

if viz_option == "Correlation Heatmap":
    st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
    
elif viz_option == "Ad Spend vs Revenue Analysis":
    st.plotly_chart(create_ad_spend_analysis(df), use_container_width=True)
    
elif viz_option == "Sales Distribution":
    st.plotly_chart(create_distribution_plot(df, 'sales_revenue_usd'), 
                   use_container_width=True)
    
elif viz_option == "Feature Distributions":
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col = st.selectbox("Select Feature", numeric_cols)
    st.plotly_chart(create_distribution_plot(df, col), use_container_width=True)
    
elif viz_option == "Scatter Plot Analysis":
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    x_col = st.selectbox("X-axis", numeric_cols)
    y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
    st.plotly_chart(create_scatter_plot(df, x_col, y_col), use_container_width=True)

# Missing values info
with st.expander("📋 Dataset Info & Missing Values"):
    st.write("**Missing Values:**")
    missing_df = df.isnull().sum().to_frame(name='Missing Count')
    missing_df['Percentage'] = (missing_df['Missing Count'] / len(df)) * 100
    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
    
    st.write("**Data Types:**")
    dtype_df = df.dtypes.to_frame(name='Data Type')
    st.dataframe(dtype_df, use_container_width=True)