#Create your own dummy set if no dataset available
#dataset hld have columns : OrderID, Product, category, Price, Quantity, Sales and Region 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Load the excel file
excel_data=pd.read_excel("SALES_DATA.xlsx")
#convert excel file into csv and json files
excel_data.to_csv("SALES_DATA.csv", index=False)
excel_data.to_json("SALES_DATA.json", orient='records', indent=4)
print("CSV and JSON files created successfully from excel!")
#Load the dataset
csv_data=pd.read_csv("SALES_DATA.csv")
json_data=pd.read_json("SALES_DATA.json")
#Explore dataset
print("Excel data:\n", excel_data.head())
print("\nCSV data:\n", csv_data.head())
print("\nJSON data:\n", json_data.head())
#Data cleaning:Combine all 3 files into 1 dataframe
all_data=pd.concat([excel_data, csv_data, json_data], ignore_index=True)
#Check for missing values
print("Missing values:", all_data.isnull().sum())
#Fill missing values (if any)
all_data['Sales'].fillna(all_data['Sales'].mean(), inplace=True)
all_data['Product'].fillna(all_data['Product'].mode()[0], inplace=True)
print(all_data)
#Check for duplicate values and remove 
all_data.drop_duplicates(inplace=True)
all_data
#Unified data shape
print("\nCombined Data:\n", all_data.shape)
#Derive variables
if 'Price' in all_data.columns and 'Quantity' in all_data.columns:
    all_data['Total Sales']=all_data['Price']*all_data['Quantity']
#Data Analysis
print("Total sales:\n", all_data['Total Sales'])
#Total sales per product
sales_by_product = all_data.groupby('Product')['Sales'].sum()
print(sales_by_product)
#Avg sales by region
if 'Region' in all_data.columns:
    avg_sales_by_region = all_data.groupby('Region')['Sales'].mean()
print(avg_sales_by_region)
#Visualise Sales per product
plt.figure(figsize=(8,4))
sales_by_product.plot(kind='bar')
plt.title("Total sales per product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.legend()
#Visualize product category distribution
if 'Category' in all_data.columns:
    category_counts = all_data['Category'].value_counts()
    category_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Product Category Distribution")
    plt.ylabel("")
    plt.show()
