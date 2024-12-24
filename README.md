# codetech-task-2

# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate a synthetic toy dataset
np.random.seed(42)

# Function to generate random reviews
def generate_review(sentiment):
    positive_words = ["amazing", "brilliant", "excellent", "fantastic", "loved", "great", "awesome", "outstanding"]
    negative_words = ["terrible", "awful", "worst", "boring", "hated", "poor", "disappointing", "bad"]

    # Choose words based on sentiment
    words = positive_words if sentiment == "positive" else negative_words
    num_words = np.random.randint(5, 15)  # Random length between 5 and 15 words
    return " ".join(np.random.choice(words, num_words))

# Create 10,000 reviews (balanced between positive and negative)
num_samples = 10000
reviews = [generate_review("positive") if i % 2 == 0 else generate_review("negative") for i in range(num_samples)]
sentiments = ["positive" if i % 2 == 0 else "negative" for i in range(num_samples)]

# Create a DataFrame
data = pd.DataFrame({"review": reviews, "sentiment": sentiments})

# Display sample data
print("Sample Data:")
print(data.head())

# Step 2: Preprocess the reviews
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

data['review'] = data['review'].apply(preprocess_text)

# Encode sentiments (positive = 1, negative = 0)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.3, random_state=42
)

# Step 4: Vectorize the text data
vectorizer = CountVectorizer(max_features=5000)  # Limit to top 5000 words
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the vectorizer and model for future use (if needed)
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'logistic_regression_model.pkl')

# Step 7: Test the model with new reviews
new_reviews = ["This movie was absolutely amazing and fantastic!",
               "I hated every moment, it was the worst experience ever.",
               "The acting was brilliant, and I loved the storyline.",
               "Poorly executed and very disappointing."]

# Preprocess and vectorize new reviews
new_reviews_preprocessed = [preprocess_text(review) for review in new_reviews]
new_reviews_vec = vectorizer.transform(new_reviews_preprocessed)

# Predict sentiments
predictions = model.predict(new_reviews_vec)
predicted_sentiments = ["positive" if pred == 1 else "negative" for pred in predictions]

# Display results
for review, sentiment in zip(new_reviews, predicted_sentiments):
    print(f"\nReview: {review}")
    print(f"Predicted Sentiment: {sentiment}")
