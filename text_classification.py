import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from datasets import load_dataset

texts = [
    "The Battle of Gettysburg was one of the most significant battles of the Civil War.",
    "The United Nations was formed in 1945 to prevent future wars and foster international cooperation.",
    "The French Revolution led to the rise of Napoleon Bonaparte and the end of the monarchy.",
    "During World War II, the Allied forces fought against the Axis powers in both Europe and the Pacific.",
    "The Vietnam War ended in 1975 with the fall of Saigon and the unification of Vietnam.",
    "The United States government enacted several economic reforms during the Great Depression.",
    "The Treaty of Versailles officially ended World War I and imposed heavy reparations on Germany.",
    "The U.S. Civil War was fought between the North and South over issues like slavery and states' rights.",
    "The Cold War between the U.S. and the Soviet Union lasted from 1947 to 1991, marked by political tension and military standoffs.",
    "The government has implemented new policies to improve public education and healthcare.",
    "In ancient Rome, the Senate was the governing body that held significant political power.",
    "The Battle of Stalingrad was a turning point in World War II, marking a major defeat for Nazi Germany.",
    "The Magna Carta was signed in 1215, limiting the power of the English monarchy and ensuring certain rights.",
    "The signing of the Declaration of Independence in 1776 marked the beginning of the United States as a sovereign nation.",
    "The Cuban Missile Crisis was a 13-day confrontation in 1962 between the U.S. and the Soviet Union over nuclear weapons.",
    "The government is currently debating the implementation of new tax reforms to address income inequality.",
    "The fall of the Berlin Wall in 1989 symbolized the end of the Cold War and the reunification of Germany.",
    "The Battle of Waterloo in 1815 marked the defeat of Napoleon Bonaparte and the end of the Napoleonic Wars.",
    "The rise and fall of the Roman Empire shaped much of Western history and culture.",
    "The American Revolution was fought from 1775 to 1783 and resulted in the independence of the United States from Britain."
]

labels = [
    0,  # War
    1,  # Government
    2,  # History
    0,  # War
    0,  # War
    1,  # Government
    0,  # War
    0,  # War
    0,  # War
    1,  # Government
    2,  # History
    0,  # War
    2,  # History
    2,  # History
    0,  # War
    1,  # Government
    2,  # History
    0,  # War
    2,  # History
    2,  # History
]


# Preprocess data (you may use a custom preprocessing function)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Bag-of-Words Model (BoW)
vectorizer = CountVectorizer(stop_words='english')  # Use TfidfVectorizer for TF-IDF features
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train logistic regression on BoW features
lr_bow = LogisticRegression(class_weight='balanced')  # Adding class weight to handle imbalances
lr_bow.fit(X_train_bow, y_train)

# Predict and evaluate
y_pred_bow = lr_bow.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)

# Calculate F1-score with zero_division=1 to avoid warnings on undefined F1-scores
f1_bow = f1_score(y_test, y_pred_bow, average='weighted', zero_division=1)

# Print the results
print("BoW Model Accuracy:", accuracy_bow)
print("BoW Model F1-score:", f1_bow)

# Load pre-trained word embeddings (example with GloVe, but you can use word2vec, etc.)
# Example: Load GloVe vectors (50-dimensional vectors in this case)
embedding_model = KeyedVectors.load_word2vec_format('./model11.bin', binary=True)
# embedding_model = Word2Vec.load("cbow_wikipedia.model").wv

def get_document_embedding(text, embedding_model):
    words = text.split()  # Tokenize text into words
    embeddings = []
    for word in words:
        if word in embedding_model:  # Access word vectors correctly
            embeddings.append(embedding_model[word])  
    if embeddings:
        return np.mean(embeddings, axis=0)  # Mean of all word embeddings
    else:
        return np.zeros(embedding_model.vector_size)  # Return zero vector if no words found

# Convert train and test texts into averaged word embeddings
X_train_embeddings = np.array([get_document_embedding(text, embedding_model) for text in X_train])
X_test_embeddings = np.array([get_document_embedding(text, embedding_model) for text in X_test])

# Train logistic regression on word embeddings
lr_embeddings = LogisticRegression(class_weight='balanced')
lr_embeddings.fit(X_train_embeddings, y_train)

# Predict and evaluate
y_pred_embeddings = lr_embeddings.predict(X_test_embeddings)
accuracy_embeddings = accuracy_score(y_test, y_pred_embeddings)
f1_embeddings = f1_score(y_test, y_pred_embeddings, average='macro')  

print("Word Embeddings Model Accuracy:", accuracy_embeddings)
print("Word Embeddings Model F1-score:", f1_embeddings)

# Compare the results
# print(f"BoW vs Word Embeddings - Accuracy: {accuracy_bow} vs {accuracy_embeddings}")
# print(f"BoW vs Word Embeddings - F1-score: {f1_bow} vs {f1_embeddings}")
