 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart_DS.csv')
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Data splited for train and test!")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}
print(f"Created {len(models)} models!")

# Step 5: Train and evaluate models
print("\n Training models...")
results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    # Use scaled data for these models
    if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }

    print(f"{name} trained! Accuracy: {results[name]['Accuracy']:.4f}")

# Step 6: Save models
print("\n Saving models...")
os.makedirs('model', exist_ok=True)

model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'kNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'XGBoost': 'xgboost.pkl'
}

for name, filename in model_files.items():
    joblib.dump(models[name], f'model/{filename}')


# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')


# Step 7: Create comparison table
print("\n Creating comparison table...")
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string())
print("="*80)

# Save to CSV
comparison_df.to_csv('model_comparison.csv')
print("\n Results saved to 'model_comparison.csv'")