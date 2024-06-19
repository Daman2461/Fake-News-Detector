import nltk
nltk.download('punkt')
nltk.download('stopwords')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
import gensim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional

# Load the data
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")
df_true['isfake'] = 0
df_fake['isfake'] = 1

df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df.drop(columns=['date'], inplace=True)
df['original'] = df['title'] + ' ' + df['text']

# Obtain additional stopwords from nltk
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Remove stopwords and words with 2 or fewer characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result

# Apply the function to the dataframe
df['clean'] = df['original'].apply(preprocess)
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

# Plot the number of samples in 'subject'
plt.figure(figsize=(8, 8))
sns.countplot(y="subject", data=df)
sns.countplot(y='isfake', data=df)

# Length of the maximum document
maxlen = max(len(nltk.word_tokenize(doc)) for doc in df.clean_joined)
print("The maximum number of words in any document is =", maxlen)

# Split data into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size=0.2)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
total_words = len(set([word for sublist in df.clean for word in sublist]))
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# Add padding
padded_train = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
padded_test = pad_sequences(test_sequences, maxlen=maxlen, truncating='post')

# Sequential Model
model = Sequential()
model.add(Embedding(total_words, output_dim=128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
y_train = np.asarray(y_train)
model.fit(padded_train, y_train, batch_size=64, validation_split=0.1, epochs=2)

# Save the model
model.save('model.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
