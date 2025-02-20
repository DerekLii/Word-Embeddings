from datasets import load_dataset
import re
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

dataset = load_dataset ("wikipedia", "20220301.simple", trust_remote_code=True)
# check the first example of the training portion of the dataset :
# print ( dataset["train"][5])

# Extract text from dataset
def extract_text(texts):
    cleaned_texts = []  # Store the cleaned texts in a list
    stop_words = set(stopwords.words('english'))  # Get the list of stopwords
    manually_removed_words = {'also', 'like', 'one', 'became', 'usually', 'could'}  # Add manually removed words
    for text in texts:
        cleaned_text = re.sub(r'\n', ' ', text).lower()  # Replace newlines with spaces Convert text to lowercase

         # Remove punctuation
        cleaned_text = ''.join([char for char in cleaned_text if char not in string.punctuation])

        # Remove stopwords
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words and word not in manually_removed_words])

        cleaned_texts.append(cleaned_text)  # Add cleaned text to the list
    return cleaned_texts

# Preprocess text for Word2Vec
def preprocess_texts(texts):
    return [simple_preprocess(text) for text in texts]

# Get cleaned and tokenized sentences

raw_texts = extract_text(dataset["train"][0:2000]["text"])
tokenized_corpus = preprocess_texts(raw_texts)

# Build and train CBOW Model (sg=0) and Skip-gram Model (sg=1)
cbow_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=3, sg=0, min_count=5, workers=4)
cbow_model.train(tokenized_corpus, total_examples=cbow_model.corpus_count, epochs=cbow_model.epochs)

skipgram_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=9, sg=1, min_count=5, workers=4)
skipgram_model.train(tokenized_corpus, total_examples=skipgram_model.corpus_count, epochs=skipgram_model.epochs)

# Save models
cbow_model.save("cbow_wikipedia.model")
skipgram_model.save("skipgram_wikipedia.model")

# Load models (if needed)
cbow_model = Word2Vec.load("cbow_wikipedia.model")
skipgram_model = Word2Vec.load("skipgram_wikipedia.model")

print("cbow_wikipedia.model saved")
print("skipgram_wikipedia.model saved")

# # Check most similar words to "earth"
# print("CBOW:", cbow_model.wv.most_similar("earth"))
# print("Skip-gram:", skipgram_model.wv.most_similar("earth"))

# # View vector representation of a word
# print(skipgram_model.wv["earth"])  # Vector embedding for "earth"