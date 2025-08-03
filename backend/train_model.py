import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1️⃣ === LOAD DATA ===

# Load Sentiment140 — limit to 50k rows
sentiment140 = pd.read_csv(
    '../data/sentiment140/training.1600000.processed.noemoticon.csv',
    encoding='latin-1',
    header=None
)
sentiment140.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
sentiment140 = sentiment140[['target', 'text']]
sentiment140 = sentiment140.sample(50000, random_state=42).reset_index(drop=True)

# Load Airline
airline = pd.read_csv('../data/twitter-airline/Tweets.csv')
airline = airline[['airline_sentiment', 'text']]
airline['target'] = airline['airline_sentiment'].map({'negative': 0, 'neutral': 2, 'positive': 4})

# Combine
combined = pd.concat([
    sentiment140[['target', 'text']],
    airline[['target', 'text']]
]).reset_index(drop=True)

print(f"Combined dataset shape: {combined.shape}")

# 2️⃣ === CLEAN TEXT ===

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

combined['text'] = combined['text'].apply(clean_text)

# Map target: 0 (negative) → 0, 2 (neutral) → 1, 4 (positive) → 2
combined['label'] = combined['target'].map({0: 0, 2: 1, 4: 2})

# 3️⃣ === SPLIT ===

X_train, X_test, y_train, y_test = train_test_split(
    combined['text'], combined['label'], test_size=0.2, random_state=42
)

# 4️⃣ === CNN ===

# Tokenize
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad
maxlen = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# Build model
cnn_model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=maxlen),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=True),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

cnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

cnn_model.summary()

# Train CNN
cnn_model.fit(X_train_pad, y_train, epochs=3, batch_size=128, validation_data=(X_test_pad, y_test))

# ✅ Ensure folders exist before saving
os.makedirs('backend/model', exist_ok=True)
os.makedirs('backend/preprocess', exist_ok=True)

# Save CNN & tokenizer
cnn_model.save('backend/model/cnn_model.h5')
joblib.dump(tokenizer, 'backend/preprocess/tokenizer.pkl')

# 5️⃣ === RANDOM FOREST ===

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_tfidf, y_train)

y_pred_rf = rf.predict(X_test_tfidf)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Save RF & vectorizer
joblib.dump(rf, 'backend/model/rf_model.pkl')
joblib.dump(vectorizer, 'backend/preprocess/tfidf_vectorizer.pkl')

print("✅ Models saved: cnn_model.h5, rf_model.pkl, tokenizer.pkl, tfidf_vectorizer.pkl")
