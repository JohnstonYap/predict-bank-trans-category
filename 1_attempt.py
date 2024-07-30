
# # Bank Transaction Category Predictive Model

# ## Import Libraries
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# ## Load and Understand Data
bank_transactions = pd.read_csv("bank_transaction.csv")
user_profiles = pd.read_csv("user_profile.csv")

print("Bank Transaction Data : \n", bank_transactions.head().to_string())
print("User Profile Data : \n", user_profiles.head().to_string())

# display information on imported data
print("Bank Transaction Info :")
print(bank_transactions.info())
print()

print("User Profile Info :")
print(user_profiles.info())

# Distribution of transaction categories
plt.figure(figsize=(12, 6))
sns.countplot(data=bank_transactions, x='category')
plt.title('Distribution of Transaction Categories')
plt.xticks(rotation=90)
plt.show()
    

# # Step 1 : Data Preprocessing
# ## Drop NaN Value

# Check the count of NaN values in each column
nan_counts = bank_transactions.isna().sum()

print("NaN counts for each column in 'bank_transactions':")
print(nan_counts)

# Show the count of rows before and after dropping
print(f"\nTotal rows before dropping: {len(bank_transactions)}")

# Drop rows with NaN values in the 'description' column
bank_transactions = bank_transactions.dropna(subset=['category'])

print(f"Total rows after dropping: {len(bank_transactions)}")

# ## Preprocess Data before Tokenization

# Function to ensure the required nltk data packages are downloaded
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

ensure_nltk_data()  

# Cached Stopwords / Pattern outside functions and reused to reduce time complexity

# Initialize the Lemmatizer and Stopwords set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Compile regex patterns
space_pattern = re.compile(r'\s+')
non_word_pattern = re.compile(r'\W+')
numeric_pattern = re.compile(r'\b\d+\b')

def preprocess_description(description):
    # Remove extra spaces and punctuation
    description = space_pattern.sub(' ', description)  # Replace multiple spaces with one
    description = non_word_pattern.sub(' ', description)  # Replace non-word characters with space
    description = numeric_pattern.sub('', description)  # Remove standalone numeric values
    description = description.lower().strip()  # Remove leading and trailing spaces
    
    # Tokenization
    tokens = word_tokenize(description)
    
    # Removing Stopwords and Lemmatization
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]   # back to its original form ( commented to maintain the features , removed to increased 2% )
    
    return ' '.join(tokens)


print("\nBefore Cleaning Bank Transaction Data : Descriptions \n", bank_transactions['description'].tail(20).to_string())

# Apply preprocessing to descriptions
bank_transactions['cleaned_description'] = bank_transactions['description'].apply(preprocess_description)

print("\nAfter Cleaning Bank Transaction Data : Descriptions\n", bank_transactions['cleaned_description'].tail(20).to_string())


# ## Feature Extraction - TF-IDF

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the cleaned descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(bank_transactions['cleaned_description'])

# The tfidf_matrix is now in sparse format and won't be converted to a dense DataFrame.
print(f"Shape of TF-IDF sparse matrix: {tfidf_matrix.shape}")
print(tfidf_matrix)

# Point out 'category' as the target variable for the prediction of transaction category
X = tfidf_matrix
y = bank_transactions['category']

# Split data into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training dataset size: {x_train.shape}")
print(f"Testing dataset size: {x_test.shape}")

# Initialize the Naive Bayes classifier
nb_model = MultinomialNB()

# Train the model using the training dataset
nb_model.fit(x_train, y_train)

# Make predictions on the test dataset
y_pred = nb_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=y.unique())

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)


