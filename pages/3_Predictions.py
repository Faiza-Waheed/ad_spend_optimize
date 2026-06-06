import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_and_prepare_data
from models.train_model import *

# Get dataset from session state
dataset_choice = st.session_state.get('dataset_choice', "Marketing & Sales Revenue Dataset (60k rows)")

# Load data
@st.cache_resource
def load_data(dataset_choice):
    df, X_train, X_test, y_train, y_test, X_future, y_future = load_and_prepare_data(dataset_choice)
    return df, X_train, X_test, y_train, y_test, X_future, y_future

df, X_train, X_test, y_train, y_test, X_future, y_future = load_data(dataset_choice)

# Retrain or get models from session state
@st.cache_resource
def get_models(dataset_choice, X_train, y_train):
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    return lr_model, rf_model, xgb_model

lr_model, rf_model, xgb_model = get_models(dataset_choice, X_train, y_train)

st.header("🔮 Interactive Sales Predictions")

st.markdown("""
Adjust the parameters below to see predicted sales revenue based on your chosen model.
This uses real-time predictions from our trained models.
""")

col1, col2 = st.columns([1, 1])

with col1:
    model_choice = st.selectbox(
        "Select Model for Prediction",
        ["XGBoost (Best Performance)", "Random Forest", "Linear Regression"]
    )
    
    # Map selection to actual model
    if "XGBoost" in model_choice:
        selected_model = xgb_model
        model_name = "XGBoost"
        # Get metrics (we need to compute or store them)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        y_pred = selected_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elif "Random Forest" in model_choice:
        selected_model = rf_model
        model_name = "Random Forest"
        y_pred = selected_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    else:
        selected_model = lr_model
        model_name = "Linear Regression"
        y_pred = selected_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

with col2:
    st.info(f"""
    **Model Selected:** {model_name}
    **Performance Metrics:**
    - R² Score: {r2:.4f}
    - RMSE: ${rmse:,.0f}
    """)

st.markdown("---")

# Dynamic input fields based on dataset
st.subheader("🎛️ Adjust Parameters")

numeric_cols = X_train.select_dtypes(include=[np.number]).columns
input_data = {}

# Create 3 columns for inputs
cols = st.columns(3)

for idx, col in enumerate(numeric_cols[:9]):  # Limit to first 9 numeric features
    with cols[idx % 3]:
        # Get reasonable min/max from training data
        min_val = float(X_train[col].min())
        max_val = float(X_train[col].max())
        mean_val = float(X_train[col].mean())
        
        input_data[col] = st.slider(
            f"{col.replace('_', ' ').title()}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100,
            format="%.2f",
            key=f"slider_{col}"
        )

# Create prediction input array
if st.button("🔮 Predict Sales Revenue", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    
    # Ensure same columns as training
    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[X_train.columns]
    
    # Make prediction
    prediction = selected_model.predict(input_df)[0]
    
    # Display prediction with animation
    st.balloons()
    
    # Create columns for better layout
    pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
    
    with pred_col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 1rem; text-align: center; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: white; margin-bottom: 0.5rem;">💰 Predicted Sales Revenue</h3>
            <h1 style="color: white; font-size: 3.5rem; margin: 0.5rem 0;">${prediction:,.2f}</h1>
            <p style="color: #e0e0e0; margin: 0;">Based on {model_name} model</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show confidence info
    with st.expander("📊 Prediction Confidence & Error Analysis"):
        st.info(f"""
        **Model Performance on Test Data:**
        - **R² Score:** {r2:.4f} (Higher is better, 1.0 = perfect)
        - **RMSE (Root Mean Square Error):** ±${rmse:,.0f}
        
        **Confidence Interval (68% confidence):**
        - Lower bound: ${prediction - rmse:,.0f}
        - Upper bound: ${prediction + rmse:,.0f}
        
        **Interpretation:** 
        There's a 68% chance the actual sales revenue will fall between 
        ${prediction - rmse:,.0f} and ${prediction + rmse:,.0f}.
        """)
    
    # Business recommendation
    st.markdown("---")
    st.subheader("💼 Business Recommendation")
    
    # Simple rule-based recommendation
    ad_spend_cols = [col for col in input_data.keys() if 'ad_spend' in col.lower()]
    total_ad_spend = sum(input_data.get(col, 0) for col in ad_spend_cols)
    
    roi_estimate = (prediction - total_ad_spend) / total_ad_spend if total_ad_spend > 0 else 0
    
    if roi_estimate > 2:
        st.success(f"✅ **Excellent ROI!** With an estimated ROI of {roi_estimate:.1%}, this budget allocation is highly efficient. Consider increasing ad spend.")
    elif roi_estimate > 1:
        st.info(f"📈 **Good ROI!** Estimated ROI of {roi_estimate:.1%}. Current allocation is working well.")
    elif roi_estimate > 0:
        st.warning(f"⚠️ **Moderate ROI.** Estimated ROI of {roi_estimate:.1%}. Consider optimizing channel mix.")
    else:
        st.error(f"❌ **Negative ROI.** Estimated ROI of {roi_estimate:.1%}. Review your ad spend strategy immediately!")

# Show sample predictions
with st.expander("📊 View Sample Predictions on Test Data"):
    sample_size = min(10, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    sample_data = X_test.iloc[sample_indices].copy()
    sample_data['Actual Revenue'] = y_test.iloc[sample_indices].values
    sample_data['Predicted Revenue'] = selected_model.predict(X_test.iloc[sample_indices])
    sample_data['Error ($)'] = sample_data['Actual Revenue'] - sample_data['Predicted Revenue']
    sample_data['Error (%)'] = (sample_data['Error ($)'] / sample_data['Actual Revenue']) * 100
    
    st.dataframe(sample_data[['Actual Revenue', 'Predicted Revenue', 'Error ($)', 'Error (%)']].style.format({
        'Actual Revenue': '${:,.0f}',
        'Predicted Revenue': '${:,.0f}',
        'Error ($)': '${:,.0f}',
        'Error (%)': '{:.1f}%'
    }), use_container_width=True)