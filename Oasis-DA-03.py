import pandas as pd

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\AB_NYC_2019.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to get an overview
print(df.head())
print(df.info())
print(df.describe(include='all'))

# Handling missing data

# Option 1: Drop rows with missing `name` and `host_name`
df = df.dropna(subset=['name', 'host_name'])

# Option 2: Impute missing `reviews_per_month` with 0 (assuming no reviews if missing)
df['reviews_per_month'].fillna(0, inplace=True)

# Convert `last_review` to datetime
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Dropping rows where `last_review` conversion failed (if any)
df = df.dropna(subset=['last_review'])

# Remove duplicates
df = df.drop_duplicates()

# Detecting outliers in the `price` column
# Define a reasonable upper limit for price
upper_limit = df['price'].quantile(0.99)

# Capping the `price` values
df['price'] = df['price'].apply(lambda x: upper_limit if x > upper_limit else x)

# Final check
print(df.info())
print(df.describe())
