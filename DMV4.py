import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1) Load dataset
path = r"/mnt/data/RealEstate_price.csv"
df = pd.read_csv(path)

# 2) Clean column names (safe normalization)
df.columns = (
    df.columns
    .str.strip()
    .str.replace(' ', '_')
    .str.replace('[^A-Za-z0-9_]+','', regex=True)
)
print("Columns after cleaning:", df.columns.tolist())

# 3) Basic type conversions & date parsing
# Convert Sales_date -> datetime (dataset uses DD-MM-YYYY)
if 'Sales_date' in df.columns:
    df['Sales_date'] = pd.to_datetime(df['Sales_date'], dayfirst=True, errors='coerce')

# 4) Missing value handling
# Price: fill with median (use parentheses — fix from original bug)
if 'Price' in df.columns:
    price_median = df['Price'].median()
    df['Price'].fillna(price_median, inplace=True)

# Bedrooms: fill with median (if present)
if 'Bedrooms' in df.columns:
    bedrooms_median = df['Bedrooms'].median()
    df['Bedrooms'].fillna(bedrooms_median, inplace=True)

# If any other specific columns need 'Unknown' fill (like location/size/society in your original),
# check existence and fill accordingly. For this dataset we don't have those exact cols.

# 5) Feature engineering
# Price per SqFt
if {'Price','SqFt'}.issubset(df.columns):
    # avoid division by zero
    df['Price_per_SqFt'] = df['Price'] / df['SqFt'].replace({0: np.nan})
    df['Price_per_SqFt'].fillna(df['Price_per_SqFt'].median(), inplace=True)

# 6) Encoding categorical columns safely
# Brick: map Yes/No -> 1/0
if 'Brick' in df.columns:
    df['Brick'] = df['Brick'].map({'Yes': 1, 'No': 0})
    # if there are other unexpected values, fill with 0 or use .fillna
    df['Brick'].fillna(0, inplace=True)

# Neighborhood: use one-hot (safer than blind label encoding)
if 'Neighborhood' in df.columns:
    df = pd.get_dummies(df, columns=['Neighborhood'], prefix='NBH', drop_first=True)

# If you really need label encoding for specific categorical variables:
# encoder = LabelEncoder()
# df['Some_col_encoded'] = encoder.fit_transform(df['Some_col'].astype(str))

# 7) Filtering (your original filtered by area_type — not present here)
# If you need to filter to a subset, adapt the condition. For now we skip that step.

# 8) Outlier capping on Price (same approach as your original but applied carefully)
if 'Price' in df.columns:
    q1 = df['Price'].quantile(0.25)
    q3 = df['Price'].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - 1.5 * iqr
    ul = q3 + 1.5 * iqr
    # Cap
    df['Price_capped'] = np.where(df['Price'] < ll, ll,
                                  np.where(df['Price'] > ul, ul, df['Price']))

# 9) Quick EDA visuals
plt.figure(figsize=(8,4))
sns.boxplot(y=df['Price'])
plt.title('Price Boxplot')
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.show()

# 10) Example aggregation: avg price by neighborhood (use columns that exist)
# If we one-hot encoded neighborhood, to show mean by original labels we can do:
if any(col.startswith('NBH_') for col in df.columns):
    # Reconstruct original NBH labels via argmax trick if needed, or simply show means by NBH columns
    nbh_cols = [c for c in df.columns if c.startswith('NBH_')]
    print("Average price by Neighborhood dummies (means for each dummy column):")
    print(df.groupby(nbh_cols)['Price'].mean().dropna(how='all'))
else:
    if 'Neighborhood' in df.columns:
        print(df.groupby('Neighborhood')['Price'].mean().sort_values(ascending=False))
