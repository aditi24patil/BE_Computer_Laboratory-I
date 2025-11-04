#ML-5
# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load dataset

df = pd.read_csv("C:\\Users\\Public\\car_evaluation.csv")

# Step 3: Add column names as per dataset documentation
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Step 4: Display basic info
print(df.head())
print("\nDataset Info:\n")
print(df.info())

# Step 5: Encode categorical features
label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
    
#ML-5
# Step 6: Split data into features and target
X = df.drop('class', axis=1)
y = df['class']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 9: Train the model
rf_model.fit(X_train, y_train)

# Step 10: Predict on test set
y_pred = rf_model.predict(X_test)

# Step 11: Evaluate model
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 12: Feature importance visualization
importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance in Random Forest Classifier")
plt.show()
