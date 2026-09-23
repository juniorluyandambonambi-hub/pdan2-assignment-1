#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset 
train_df = pd.read_csv(r"C:\Users\luyanda\Downloads\train.csv")
test_df = pd.read_csv(r"C:\Users\luyanda\Downloads\test.csv")

# 1. Check for missing values in the training data
print("Missing values before cleaning:")
print(train_df.isnull().sum())

# 2. Drop missing values (if any)
train_df.dropna(inplace=True)

# 3. Clean the text (lowercase)
# This ensures that "The" and "the" are treated as the same word.
train_df['text'] = train_df['text'].apply(lambda x: x.lower())
test_df['text'] = test_df['text'].apply(lambda x: x.lower())

# 4. Visualize the number of texts per author
# This helps us see if the dataset is balanced.
plt.figure(figsize=(8, 5))
sns.countplot(x='author', data=train_df)
plt.title('Number of Texts per Author')
plt.xlabel('Author')
plt.ylabel('Count')
plt.show()

# 5. Visualize the length of the texts
# This helps us decide the MAX_LEN for padding.
train_df['text_length'] = train_df['text'].apply(len)
plt.figure(figsize=(8, 5))
sns.histplot(train_df['text_length'], bins=50, kde=True)
plt.title('Distribution of Text Lengths')
plt.xlabel('Length of Text')
plt.show()


# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Separate features (X) and labels (y)
X_text = train_df['text'].values
y_labels = train_df['author'].values

# 2. Encode the labels (EAP, HPL, MWS -> 0, 1, 2)
# The model cannot understand text labels, so we convert them to numbers.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)

# 3. Tokenize the text (Turn words into numbers)
VOCAB_SIZE = 10000 # Only use the top 10,000 words
MAX_LEN = 100 # Make all texts 100 words long (adjust based on EDA)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)
sequences = tokenizer.texts_to_sequences(X_text)

# 4. Pad the sequences (Make them all the same length)
# This is required for the LSTM to process the data.
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# 5. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# In[6]:


get_ipython().system('pip install tensorflow')


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# 1. Build the LSTM Model
model = Sequential([
    # Embedding layer: turns numbers into dense vectors
    Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN),

    # Bidirectional LSTM: reads text forwards and backwards
    # This helps the model understand context better.
    Bidirectional(LSTM(64, return_sequences=False)),

    # Dropout layer: prevents overfitting (memorizing)
    Dropout(0.5),

    # Dense layer: learns complex patterns
    Dense(64, activation='relu'),

    # Output layer: gives probability for each author
    # Softmax is used for multi-class classification.
    Dense(num_classes, activation='softmax')
])

# 2. Compile the model

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. Show model summary
model.summary()

# 4. Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# In[1]:


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Make predictions on the test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Print Classification Report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 3. Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 4. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Author')
plt.ylabel('Actual Author')
plt.show()


# In[11]:


from tensorflow.keras.callbacks import EarlyStopping

# 1. Build a simpler model to prevent overfitting
model_retrained = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_LEN),
    LSTM(32), # Reduced from 64 to 32
    Dropout(0.5), # Increased dropout
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model_retrained.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 2. Add Early Stopping
# This stops training if the validation loss doesn't improve for 3 epochs.
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 3. Retrain the model
history_retrained = model_retrained.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)


# In[ ]:





# In[ ]:




