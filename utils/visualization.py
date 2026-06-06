import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        title="Feature Correlations"
    )
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """Create interactive scatter plot"""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=f"{x_col} vs {y_col}",
        trendline="ols",
        opacity=0.6
    )
    return fig

def create_distribution_plot(df, column):
    """Create distribution histogram"""
    fig = px.histogram(
        df, x=column, nbins=50,
        title=f"Distribution of {column}",
        marginal="box"
    )
    return fig

def create_ad_spend_analysis(df):
    """Create ad spend vs revenue analysis"""
    fig = go.Figure()
    
    # Find ad spend columns
    ad_cols = [col for col in df.columns if 'ad_spend' in col.lower() or 'tv' in col.lower() or 'radio' in col.lower()]
    
    for col in ad_cols:
        if col in df.columns and col != 'sales_revenue_usd':
            fig.add_trace(go.Scatter(
                x=df[col], y=df['sales_revenue_usd'],
                mode='markers',
                name=col,
                opacity=0.5,
                marker=dict(size=5)
            ))
    
    fig.update_layout(
        title="Ad Spend vs Sales Revenue",
        xaxis_title="Ad Spend (USD)",
        yaxis_title="Sales Revenue (USD)",
        hovermode='closest'
    )
    return fig

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        # For linear regression, take absolute values
        if len(importance.shape) > 1:
            importance = importance[0]
    else:
        return None
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig = px.bar(
        feature_importance, x='importance', y='feature',
        orientation='h', title=f"Top {top_n} Feature Importance",
        color='importance', color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_predictions_vs_actual(y_test, y_pred, model_name):
    """Plot predictions vs actual values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=8, opacity=0.6)
    ))
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{model_name}: Predictions vs Actual",
        xaxis_title="Actual Sales Revenue (USD)",
        yaxis_title="Predicted Sales Revenue (USD)",
        showlegend=True
    )
    return fig