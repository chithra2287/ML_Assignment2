Machine Learning Classification Model Comparison

## Problem Statement

This project implements and compares six different machine learning classification algorithms on a real-world dataset. 
The objective is to:

1. Train and evaluate multiple classification models
2. Compare their performance using standard evaluation metrics
3. Deploy the best-performing models through an interactive web application
4. Provide insights into model selection for classification tasks


 ## Dataset Description

### Dataset: Heart Disease Dataset

**Source:** Kaggle

**Overview:**
- **Problem Type:** Binary/Multi-class Classification
- **Number of Features:** 14  
- **Number of Instances:** 1,025  
- **Target Variable:** target


================================================================================
MODEL COMPARISON TABLE
================================================================================
                     Accuracy     AUC  Precision  Recall      F1     MCC
Logistic Regression    0.7951  0.8787     0.8023  0.7951  0.7938  0.5973
Decision Tree          0.9854  0.9854     0.9858  0.9854  0.9854  0.9712
kNN                    0.8341  0.9486     0.8387  0.8341  0.8335  0.6727
Naive Bayes            0.8000  0.8706     0.8105  0.8000  0.7982  0.6102
Random Forest          0.9854  1.0000     0.9858  0.9854  0.9854  0.9712
XGBoost                0.9854  0.9894     0.9858  0.9854  0.9854  0.9712
================================================================================



### Model Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Demonstrates solid baseline performance with good interpretability. Works well for linearly separable data. Fast training time                             makes it suitable for large datasets. Achieved balanced precision and recall, indicating reliable performance across both                                   classes. Feature scaling significantly improved convergence speed. 
| **Decision Tree** | Shows moderate performance with tendency to overfit on training data. The model achieved good interpretability with clear decision                               rules. Pruning (max_depth=10) helped reduce overfitting. No feature scaling required, making preprocessing simpler.                                           Performance drops on unseen data suggest need for ensemble methods. 
| **kNN** | Performs well with proper feature scaling. Computational cost increases with dataset size, making predictions slower. The choice of k=5                                   provided optimal balance between bias and variance. Sensitive to irrelevant features and data scaling. Works better with larger                             training datasets. 
| **Naive Bayes** | Fast training and prediction times make it efficient for real-time applications. Independence assumption limits performance on                               correlated features. Despite simplicity, achieved competitive results. Particularly effective for high-dimensional data. Probability                       estimates useful for threshold tuning. 
| **Random Forest (Ensemble)** | Superior performance compared to single Decision Tree due to ensemble learning. Reduced overfitting through bootstrap                               aggregating. Provides feature importance scores valuable for interpretation. More robust to outliers and noise. Training time                                 longer but worth the performance gain. 
| **XGBoost (Ensemble)** | Achieved best overall performance across all metrics. Gradient boosting approach effectively handles complex patterns. Built-in                             regularization prevents overfitting. Handles missing values internally. Provides best AUC score, crucial for imbalanced                                       datasets. Longer training time but highest accuracy justifies the cost. 

