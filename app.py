import streamlit as st
from models.train_model import load_pretrained_models

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Ad Spend Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=80)
    st.title("📊 Navigation")
    
    # Dataset selector (global state)
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Marketing & Sales Revenue Dataset (60k rows)", 
         "Simplified Advertising Dataset (20k rows)"]
    )
    
    # Store in session state for other pages
    st.session_state['dataset_choice'] = dataset_choice
    
    # Load pre-trained models if available
    if 'pretrained_models' not in st.session_state:
        with st.spinner("Loading pre-trained models..."):
            model_data = load_pretrained_models(dataset_choice)
            if model_data:
                st.session_state['pretrained_models'] = model_data
                st.session_state['models_loaded'] = True
                st.success("✅ Models loaded from cache!")
            else:
                st.session_state['models_loaded'] = False
                st.warning("⚠️ No pre-trained models found. Models will be trained on first use.")
    
    st.markdown("---")
    st.info(
        """
        **About this Dashboard:**
        - Compare ML models for sales prediction
        - Optimize ad spend allocation
        - Get future revenue predictions
        
        **Models Available:**
        - Linear Regression
        - Random Forest
        - XGBoost
        """
    )

# Main header
st.markdown('<h1 class="main-header">🎯 Ad Spend Optimization & Sales Performance Analysis</h1>', 
            unsafe_allow_html=True)

# Navigation - This creates the page links
page = st.navigation([
    st.Page("pages/1_Data_Exploration.py", title=" Data Exploration", icon="📈"),
    st.Page("pages/2_Model_Comparison.py", title=" Model Comparison", icon="🤖"),
    st.Page("pages/3_Predictions.py", title=" Predictions", icon="🎯"),
    st.Page("pages/4_Final_Analysis.py", title=" Final Analysis", icon="💡"),
])

# Run the selected page
page.run()
