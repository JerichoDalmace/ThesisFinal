import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Data preprocessing function
def preprocess_text(text):
    text = re.sub(r'[\U0001F600-\U0001F64F'  # Emoticons
                  r'\U0001F300-\U0001F5FF'  # Symbols & pictographs
                  r'\U0001F680-\U0001F6FF'  # Transport & map symbols
                  r'\U0001F1E0-\U0001F1FF'  # Flags
                  r'\U00002702-\U000027B0'  # Dingbats
                  r'\U000024C2-\U0001F251]', '', text, flags=re.UNICODE)
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Load dataset
data = pd.read_csv('Train.csv')

# Preprocess reviews
data['review'] = data['review'].fillna('').apply(preprocess_text)
data.drop_duplicates(subset=['review'], inplace=True)

# Check for missing labels and drop them
data = data[data['sentiment'].notna()]

# Ask for user input to specify train-test split percentage
test_size_percentage = float(input("Enter the percentage of data to be used for testing (0-1): "))

# Ensure the input is between 0 and 1
if test_size_percentage < 0 or test_size_percentage > 1:
    raise ValueError("Test size must be between 0 and 1.")

# Split data based on user input
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=test_size_percentage, random_state=42)

# Model pipeline with TF-IDF
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate model
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))

# Mapping sentiment values to labels
sentiment_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
y_test_labels = [sentiment_map[s] for s in y_test]
predicted_labels = [sentiment_map[s] for s in predicted]

# Generate confusion matrix with mapped labels
cm = confusion_matrix(y_test_labels, predicted_labels, labels=['negative', 'neutral', 'positive'])

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.show()

# Plot distribution of actual vs predicted sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=data.loc[X_test.index], palette="Blues", alpha=0.7, label="Actual", hue='sentiment', hue_order=['negative', 'neutral', 'positive'])
sns.countplot(x=predicted_labels, palette="Reds", alpha=0.7, label="Predicted", hue_order=['negative', 'neutral', 'positive'])
plt.title("Sentiment Distribution: Actual vs Predicted")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend()
plt.show()

# Save model pipeline
joblib.dump(model, 'naive_bayes_mark3.pkl')

print(f"Model saved as 'naive_bayes_mark3.pkl' with test size {test_size_percentage*100}%")
