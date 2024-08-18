import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\apps.csv"
apps_df = pd.read_csv(file_path)

# Data Cleaning and Preparation
# Remove any duplicates
apps_df.drop_duplicates(subset='App', keep='first', inplace=True)

# Convert 'Installs' to a numerical value
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '').str.replace('+', '').astype(int)

# Convert 'Price' to a numerical value
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)

# Handle missing values
apps_df['Rating'].fillna(apps_df['Rating'].mean(), inplace=True)
apps_df['Size'].fillna(apps_df['Size'].mean(), inplace=True)

# Convert 'Last Updated' to datetime
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'])

# Categorizing Apps based on Category
category_count = apps_df['Category'].value_counts()

# Metrics Analysis
# Analyzing App Ratings
plt.figure(figsize=(10, 6))
sns.histplot(apps_df['Rating'], bins=20, kde=True, color='blue')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Analyzing App Size vs. Installs
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size', y='Installs', hue='Category', data=apps_df)
plt.title('App Size vs. Installs')
plt.xlabel('Size (MB)')
plt.ylabel('Installs')
plt.show()

# Analyzing Pricing Trends
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Price', data=apps_df)
plt.title('App Pricing Trends by Category')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Price ($)')
plt.show()

# Sentiment Analysis (Assuming we have a 'Sentiment' column)
# Note: This dataset doesn't have a 'Sentiment' column, but here's how you would do it if it existed
# apps_df['Sentiment'] = apps_df['Reviews'].apply(lambda x: 'Positive' if x > 50000 else 'Negative')
# sentiment_count = apps_df['Sentiment'].value_counts()

# Visualizing the distribution of app categories
plt.figure(figsize=(10, 6))
category_count.plot(kind='bar')
plt.title('App Distribution Across Categories')
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.show()

# Interactive Visualizations
import plotly.express as px

# Interactive Scatter Plot for Size vs Installs with Category
fig = px.scatter(apps_df, x='Size', y='Installs', color='Category', size='Reviews', hover_name='App', title='App Size vs. Installs (Interactive)')
fig.show()

# End of the code
