import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\Housing.csv"  # Update this with the correct file path
housing_data = pd.read_csv(file_path)

# Convert categorical variables to numerical (using one-hot encoding)
housing_data_encoded = pd.get_dummies(housing_data, drop_first=True)

# Define the features (X) and the target variable (y)
X = housing_data_encoded.drop('price', axis=1)
y = housing_data_encoded['price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Optionally, visualize the relationship between actual and predicted prices
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
