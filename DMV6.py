import pandas as pd
import matplotlib.pyplot as plt

# 1. Import dataset
df = pd.read_csv("C:/Users/User 58/Downloads/DMV6.csv")

# 2. Explore dataset
print(df.head())
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# 3. Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True)

# 4. Aggregate total sales by region
sales_by_region = df.groupby('Region')['Sales_Amount'].sum().reset_index().sort_values(by='Sales_Amount', ascending=False)
print("\nTotal Sales by Region:\n", sales_by_region)

# 5. Bar plot – Sales distribution by region
plt.figure(figsize=(8,4))
plt.bar(sales_by_region['Region'], sales_by_region['Sales_Amount'], color='skyblue')
plt.title("Total Sales Amount by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=45)
plt.show()

# 6. Identify top performing region
top_region = sales_by_region.iloc[0]
print(f"\n🏆 Top Performing Region: {top_region['Region']} with sales of {top_region['Sales_Amount']:.2f}")

# 7. Group by Region + Product Category
sales_region_product = df.groupby(['Region', 'Product_category'])['Sales_Amount'].sum().reset_index()
print("\nSales by Region and Product Category:\n", sales_region_product)

# 8. Stacked Bar Plot – Region vs Product Category
pivot_sales = sales_region_product.pivot(index='Region', columns='Product_category', values='Sales_Amount')

pivot_sales.plot(kind='bar', stacked=True, figsize=(10,5))
plt.title("Sales Amount by Region and Product Category (Stacked Bar)")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.legend(title="Product Category")
plt.show()
