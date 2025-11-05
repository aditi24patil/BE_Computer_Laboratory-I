# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('Telecom_Customer_Churn.csv')

# 2. Explore the dataset
print("First 5 rows:\n", df.head())
print("\nDataset Information:\n")
df.info()
print("\nMissing values:\n", df.isnull().sum())

# 3. Handle Missing Values (if any appear)
# TotalCharges might look numeric, but could be object due to spaces or missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# For any remaining missing values
df.fillna(method='ffill', inplace=True)

# 4. Remove duplicate records
df.drop_duplicates(inplace=True)

# 5. Standardize inconsistent data
# Convert all yes/no fields to lowercase and unify them
yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_cols:
    df[col] = df[col].str.strip().str.lower().map({'yes': 1, 'no': 0, 'no phone service': 0})

# 6. Ensure Correct Data Types
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# 7. Handle Outliers (example: MonthlyCharges, TotalCharges capped at 99th percentile)
for col in ['MonthlyCharges', 'TotalCharges']:
    upper_limit = df[col].quantile(0.99)
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

# 8. Feature Engineering
# Create a new feature: Tenure Group
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], 
                            labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr'])

# 9. Scaling numerical data
scaler = StandardScaler()
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_features] = scaler.fit_transform(df[num_features])

# 10. Train-test split
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes after split: ")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train:", y_train.shape)
print("y_test: ", y_test.shape)

# 11. Save cleaned dataset
df.to_csv('Telecom_Customer_Churn_Cleaned.csv', index=False)
print("\n✅ Cleaned dataset exported as 'Telecom_Customer_Churn_Cleaned.csv'.")
