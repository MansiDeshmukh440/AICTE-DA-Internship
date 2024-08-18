import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\retail_sales_dataset.csv"

data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Data Preview:")
print(data.head())

# Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill or drop missing values as needed (example: dropping rows with missing values)
data = data.dropna()

# Convert date columns to datetime format if applicable
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Time Series Analysis: Analyzing sales trends over time
if 'Date' in data.columns and 'Sales' in data.columns:
    time_series_data = data.groupby('Date')['Sales'].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_series_data['Date'], time_series_data['Sales'], marker='o')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Customer Analysis: Analyzing customer demographics and purchasing behavior
if 'Customer_ID' in data.columns:
    customer_data = data.groupby('Customer_ID').agg({'Sales': 'sum', 'Product_ID': 'count'}).reset_index()
    customer_data.columns = ['Customer_ID', 'Total_Sales', 'Total_Products']
    
    plt.figure(figsize=(14, 6))
    sns.histplot(customer_data['Total_Sales'], kde=True)
    plt.title('Distribution of Total Sales per Customer')
    plt.xlabel('Total Sales')
    plt.ylabel('Frequency')
    plt.show()

# Product Analysis: Analyzing product sales and popularity
if 'Product_ID' in data.columns:
    product_data = data.groupby('Product_ID').agg({'Sales': 'sum', 'Customer_ID': 'count'}).reset_index()
    product_data.columns = ['Product_ID', 'Total_Sales', 'Total_Customers']
    
    plt.figure(figsize=(14, 6))
    top_products = product_data.sort_values(by='Total_Sales', ascending=False).head(10)
    sns.barplot(x='Product_ID', y='Total_Sales', data=top_products)
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Product ID')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

# Heatmap: Correlation between different numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Recommendations based on EDA
print("\nRecommendations:")
print("1. Focus on promoting the top-performing products to maximize sales.")
print("2. Target high-value customers with personalized marketing campaigns.")
print("3. Monitor sales trends to anticipate peak periods and manage inventory effectively.")
print("4. Explore cross-selling opportunities based on customer purchasing behavior.")
