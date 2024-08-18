import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Step 1: Load the Dataset
file_path = "C:\\Users\\Acer\\Desktop\\Game\\Twitter_Data.csv"  # Update with the correct path
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Drop rows with missing values
df.dropna(inplace=True)

# Convert 'category' to integer type
df['category'] = df['category'].astype(int)

# Step 3: Feature Engineering and Model Selection
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)

# Creating a pipeline that includes TF-IDF vectorization and Naive Bayes classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# Step 4: Model Training
pipeline.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = pipeline.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
