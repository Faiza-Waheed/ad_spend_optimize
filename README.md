# 🎯 Ad Spend Optimization & Sales Performance Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache-green.svg)](LICENSE)

An interactive machine learning dashboard for optimizing advertising spend and predicting sales revenue. Compare multiple ML models, visualize performance metrics, and get data-driven recommendations for budget allocation.

## 🚀 Live Demo

[View the live application](https://your-app-url.streamlit.app) *(Update with your actual Streamlit Cloud URL)*

## 📊 Features

### 🔍 Data Exploration
- Interactive dataset selection (2 datasets available)
- Correlation heatmaps and distribution analysis
- Ad spend vs revenue visualizations
- Missing value analysis

### 🤖 Model Comparison
- **3 ML Models**: Linear Regression, Random Forest, XGBoost
- Performance metrics: R² Score, RMSE, MAE, MAPE
- Feature importance analysis
- Predictions vs actual visualizations

### 🔮 Interactive Predictions
- Real-time sales prediction with adjustable parameters
- Model selection (choose best performer)
- Confidence intervals and error analysis
- ROI estimation and business recommendations

### 💡 Business Insights
- ROI analysis by channel
- Data-driven optimization recommendations
- Future performance predictions on unseen data
- Executive summary with key KPIs

## 📁 Dataset Options

### 1. Marketing & Sales Revenue Dataset (60k rows)
- **Features**: Online/offline ad spend, marketing budget, discounts, promotions
- **Categorical**: Region, sales channel, product category, customer segment, season
- **Target**: Sales revenue (USD)
- **Missing Values**: ~3% (realistic business data)

### 2. Simplified Advertising Dataset (20k rows)
- **Features**: TV, radio, newspaper ad spend
- **Target**: Sales revenue (USD)
- **Use Case**: Quick testing and fundamentals demonstration

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn, XGBoost, Random Forest
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Cloud

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ad-spend-optimizer.git
cd ad-spend-optimizer



ad-spend-optimizer/
├── app.py                          # Main application entry point
├── pages/                          # Multi-page app sections
│   ├── 1_Data_Exploration.py      # Data visualization tab
│   ├── 2_Model_Comparison.py      # Model performance comparison
│   ├── 3_Predictions.py           # Interactive predictions
│   └── 4_Final_Analysis.py        # Business insights & recommendations
├── utils/                          # Utility functions
│   ├── data_loader.py             # Data generation & loading
│   └── visualization.py           # Plotting functions
├── models/                         # ML model code
│   └── train_model.py             # Model training & evaluation
├── scripts/                        # Helper scripts
│   └── train_and_save_models.py   # Offline model training
├── saved_models/                   # Pre-trained models (generated)
│   ├── *_models.pkl               # Serialized models
│   └── *_test_data.pkl            # Test/future splits
├── requirements.txt                # Python dependencies
└── README.md                       # This file
