import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\WineQT.csv"  # Update this with the correct file path
wine_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(wine_data.head())

# Check for missing values
print(wine_data.isnull().sum())

# Define features (X) and target (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
rf_model = RandomForestClassifier(random_state=42)
sgd_model = SGDClassifier(random_state=42)
svc_model = SVC(random_state=42)

# Train and evaluate the Random Forest model
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Classifier")
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))

# Train and evaluate the SGD model
sgd_model.fit(X_train, y_train)
sgd_y_pred = sgd_model.predict(X_test)
print("Stochastic Gradient Descent Classifier")
print("Accuracy:", accuracy_score(y_test, sgd_y_pred))
print(classification_report(y_test, sgd_y_pred))

# Train and evaluate the SVC model
svc_model.fit(X_train, y_train)
svc_y_pred = svc_model.predict(X_test)
print("Support Vector Classifier")
print("Accuracy:", accuracy_score(y_test, svc_y_pred))
print(classification_report(y_test, svc_y_pred))

# Confusion matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion matrix for SGD
sgd_cm = confusion_matrix(y_test, sgd_y_pred)
sns.heatmap(sgd_cm, annot=True, fmt='d', cmap='Blues')
plt.title("SGD Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion matrix for SVC
svc_cm = confusion_matrix(y_test, svc_y_pred)
sns.heatmap(svc_cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVC Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
