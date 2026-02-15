 

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


# Page configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)
st.title("ML Assignment 2")


# Title and description

st.subheader("Implementation of Multiple Classification Models")
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("⚙️ Configuration")

# Model selection dropdown
model_options = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbors': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys())
)

# Information section
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload your test dataset (CSV)
2. Select a model from dropdown
3. View evaluation metrics
4. Analyze confusion matrix
""")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
     
    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing test data"
    )

with col2:
     
    st.info(f"**Selected Model:**\n\n{selected_model_name}")

# Function to load model
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to load scaler
@st.cache_resource
def load_scaler():
    """Load the scaler"""
    try:
        scaler_path = 'model/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            return scaler
        else:
            return None
    except:
        return None

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC Score
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
            else:  # Multi-class
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes=None):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Count'},
                xticklabels=classes if classes else 'auto',
                yticklabels=classes if classes else 'auto')
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    return fig

# Process uploaded file
if uploaded_file is not None:
    
    try:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        
        # Display dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset statistics
        with st.expander("📊 Dataset Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
        
        # Assume last column is target
        st.markdown("---")
        
        target_column = st.selectbox(
            "Select Target Column",
            df.columns.tolist(),
            index=len(df.columns) - 1
        )
        
        if st.button("🚀 Run Prediction", type="primary"):
            
            with st.spinner("Training and evaluating model..."):
                
                # Separate features and target
                X = df.drop(target_column, axis=1)
                y = df[target_column]
                
                # Load model
                model = load_model(model_options[selected_model_name])
                
                if model is not None:
                    
                    # Load scaler if needed
                    scaler = load_scaler()
                    
                    # Apply scaling for specific models
                    if selected_model_name in ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes']:
                        if scaler is not None:
                            X_processed = scaler.transform(X)
                        else:
                            # Create temporary scaler if not available
                            temp_scaler = StandardScaler()
                            X_processed = temp_scaler.fit_transform(X)
                    else:
                        X_processed = X
                    
                    # Make predictions
                    y_pred = model.predict(X_processed)
                    
                    # Get probability predictions if available
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_processed)
                        if len(np.unique(y)) == 2:
                            y_pred_proba = y_pred_proba[:, 1]
                    else:
                        y_pred_proba = None
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y, y_pred, y_pred_proba)
                    
                    st.markdown("---")
                    
                    # Display metrics
                    st.subheader("📊 Evaluation Metrics")
                    
                    # Create 3 columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    
                    with col2:
                        st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    
                    with col3:
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                        st.metric("MCC Score", f"{metrics['MCC']:.4f}")
                    
                    st.markdown("---")
                    
                    # Display confusion matrix and classification report
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔲 Confusion Matrix")
                        cm = confusion_matrix(y, y_pred)
                        fig = plot_confusion_matrix(cm, classes=np.unique(y).tolist())
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("📄 Classification Report")
                        report = classification_report(y, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), use_container_width=True)
                    
                    # Download predictions
                    st.markdown("---")
                    st.subheader("💾 Download Predictions")
                    
                    result_df = df.copy()
                    result_df['Predicted'] = y_pred
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Prediction completed successfully!")
                
                else:
                    st.error("❌ Could not load the model. Please check if model files exist.")
        
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        st.info("Please ensure your CSV is properly formatted and contains the correct columns.")

else:
    # Display welcome message
    st.info("👆 Please upload a CSV file to begin")
    
    # Display model comparison table if available
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison")
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.8500, 0.8200, 0.8400, 0.8100, 0.8800, 0.9000],
        'AUC': [0.8900, 0.8500, 0.8700, 0.8400, 0.9200, 0.9300],
        'Precision': [0.8400, 0.8100, 0.8300, 0.8000, 0.8700, 0.8900],
        'Recall': [0.8500, 0.8200, 0.8400, 0.8100, 0.8800, 0.9000],
        'F1': [0.8450, 0.8150, 0.8350, 0.8050, 0.8750, 0.8950],
        'MCC': [0.6800, 0.6300, 0.6700, 0.6100, 0.7500, 0.7900]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.caption("*Note: These are example metrics. Actual values will be displayed after prediction.*")

# Footer
st.markdown("---")
 
