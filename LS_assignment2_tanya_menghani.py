### Question  1



import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api


df = pd.read_csv('spam.csv')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

def preprocess_text(text):

    text = text.lower()
    #urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and numbers, keep only letters and basic punctuation
    text = re.sub(r'[^a-zA-Z\s\'\-]', '', text)
    
    # Remove common stopwords using regex patterns
    # Pattern matches: pronouns, articles, conjunctions, prepositions, etc.
    stopword_pattern = re.compile(
        r'\b(i|me|my|myself|we|our|ours|ourselves|you|your|yours|'
        r'yourself|yourselves|he|him|his|himself|she|her|hers|'
        r'herself|it|its|itself|they|them|their|theirs|themselves|'
        r'what|which|who|whom|this|that|these|those|am|is|are|was|'
        r'were|be|been|being|have|has|had|having|do|does|did|doing|'
        r'a|an|the|and|but|if|or|because|as|until|while|of|at|by|'
        r'for|with|about|against|between|into|through|during|before|'
        r'after|above|below|to|from|up|down|in|out|on|off|over|under|'
        r'again|further|then|once|here|there|when|where|why|how|all|'
        r'any|both|each|few|more|most|other|some|such|no|nor|not|only|'
        r'own|same|so|than|too|very|s|t|can|will|just|don|should|now|'
        r'\d+)\b', flags=re.IGNORECASE)
    
    text = stopword_pattern.sub('', text)
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 1]
    
    return tokens

df['processed'] = df['message'].apply(preprocess_text)

# Load Word2Vec model

w2v_model = api.load('word2vec-google-news-300')

# Convert messages to vectors by averaging word vectors
def message_to_vector(tokens, model):
    vectors = []
    for word in tokens:
        try:
            vectors.append(model[word])
        except KeyError:
            continue   #ignoring words not in model's vocab
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['vector'] = df['processed'].apply(lambda x: message_to_vector(x, w2v_model))

# Prepare data for training
X = np.stack(df['vector'].values)
y = df['label'].map({'ham': 0, 'spam': 1}).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction function
def predict_message_class(model, w2v_model, message):
    tokens = preprocess_text(message)
    vector = message_to_vector(tokens, w2v_model).reshape(1, -1)
    prediction = model.predict(vector)[0]
    return 'spam' if prediction == 1 else 'ham'



### Question 2


import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api

# Downloading NLTK resources for lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# Loading dataset 
df = pd.read_csv('tweets.csv')
df = df[['airline_sentiment', 'text']]  


# Contraction mapping
CONTRACTIONS = {
    r"won't": "will not",
    r"can't": "cannot",
    r"n't": " not",
    r"'re": " are",
    r"'s": " is",
    r"'d": " would",
    r"'ll": " will",
    r"'t": " not",
    r"'ve": " have",
    r"'m": " am"
}

def preprocess_tweet(tweet):
    #  lowercase
    tweet = str(tweet).lower()
    
    # Removing URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    tweet = re.sub(r'@\w+|#\w+', '', tweet)
    
    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        tweet = re.sub(contraction, expansion, tweet)
    
    # Remove punctuation (keep apostrophes for contractions)
    tweet = re.sub(r'[^\w\s\']', '', tweet)
    
    # Remove remaining special characters and numbers
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    
    # Tokenize
    words = tweet.split()
    
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words

# Apply preprocessing
df['processed'] = df['text'].apply(preprocess_tweet)

# Load Word2Vec model
w2v_model = api.load('word2vec-google-news-300')

def tweet_to_vector(tokens, model):
    vectors = []
    for word in tokens:
        try:
            vectors.append(model[word])
        except KeyError:
            continue # ignoring those not in vocab
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

df['vector'] = df['processed'].apply(lambda x: tweet_to_vector(x, w2v_model))
X = np.stack(df['vector'].values)
y = df['airline_sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
lr_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_model.fit(X_train, y_train)

# Evaluate
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction function
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_tweet(tweet)
    vector = tweet_to_vector(tokens, w2v_model).reshape(1, -1)
    pred_num = model.predict(vector)[0]
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_map[pred_num]
