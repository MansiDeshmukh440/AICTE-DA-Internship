import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\creditcard.csv"  # Update this path if necessary
df = pd.read_csv(file_path)

# 1. Exploratory Data Analysis (EDA)
print("Checking for missing values:")
print(df.isnull().sum())  # Check for missing values

print("\nSummary statistics:")
print(df.describe())  # Summary statistics

print("\nDistribution of the 'Class' column (fraud vs. non-fraud):")
print(df['Class'].value_counts())  # Distribution of the target variable

# 2. Visualizing the Data
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of the Features')
plt.show()

# 3. Feature Engineering
df['Log_Amount'] = df['Amount'].apply(lambda x: np.log(x + 1))  # Log transformation of 'Amount'
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)  # Extract hour from 'Time'

print("\nSample of engineered features:")
print(df[['Time', 'Hour', 'Amount', 'Log_Amount']].head())

# 4. Model Building
X = df.drop(columns=['Class'])  # All columns except the target 'Class'
y = df['Class']  # The target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
