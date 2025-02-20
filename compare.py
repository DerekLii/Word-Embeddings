from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors

# Load pretrained model 1
pretrained_model_1_path = './model11.bin'
pretrained_model_1 = KeyedVectors.load_word2vec_format(pretrained_model_1_path, binary=True)
# unique_words = pretrained_model_1.index_to_key


# print(f"Total unique words: {len(unique_words)}")
# print("First 100 unique words:", unique_words[:100])

# # Load pretrained model 2
pretrained_model_2_path = './model12.bin'
pretrained_model_2 = KeyedVectors.load_word2vec_format(pretrained_model_2_path, binary=True)
# unique_words = pretrained_model_2.index_to_key


# print(f"Total unique words: {len(unique_words)}")
# print("First 100 unique words:", unique_words[:100])


# Load models (if needed)
cbow_model = Word2Vec.load("cbow_wikipedia.model")
skipgram_model = Word2Vec.load("skipgram_wikipedia.model")

# vocab = cbow_model.wv.index_to_key  # List of words in the vocabulary

# # Get word frequencies and sort them in descending order
# word_freqs = [(word, cbow_model.wv.get_vecattr(word, 'count')) for word in vocab]
# sorted_word_freqs = sorted(word_freqs, key=lambda x: x[1], reverse=True)

# # Print the top 10 most frequent words
# print("Top 100 most frequent words:")
# for word, freq in sorted_word_freqs[:100]:
#     print(f"{word}: {freq}")

print("---------Query 1: april + march - month---------\n")

result = cbow_model.wv.most_similar(positive=["april", "march"], negative=["month"], topn=10)
print("CBOW:", result, "\n")

result = skipgram_model.wv.most_similar(positive=["april", "march"], negative=["month"], topn=10)
print("skipgram:", result, "\n")

result = pretrained_model_1.most_similar(positive=["april", "march"], negative=["month"], topn=10)
print("Pretrained1:", result, "\n")

result = pretrained_model_2.most_similar(positive=["april", "march"], negative=["month"], topn=10)
print("Pretrained2:", result, "\n")

print("---------Query 2: people + government - war---------\n")

result = cbow_model.wv.most_similar(positive=["people", "government"], negative=["war"], topn=10)
print("CBOW:", result, "\n")

result = skipgram_model.wv.most_similar(positive=["people", "government"], negative=["war"], topn=10)
print("skipgram:", result, "\n")

result = pretrained_model_1.most_similar(positive=["people", "government"], negative=["war"], topn=10)
print("Pretrained1:", result, "\n")

result = pretrained_model_2.most_similar(positive=["people", "government"], negative=["war"], topn=10)
print("Pretrained2:", result, "\n")

print("---------Query 3: war---------\n")

result = cbow_model.wv.most_similar(positive=["war"], topn=10)
print("CBOW:", result, "\n")

result = skipgram_model.wv.most_similar(positive=["war"], topn=10)
print("skipgram:", result, "\n")

result = pretrained_model_1.most_similar(positive=["war"], topn=10)
print("Pretrained1:", result, "\n")

result = pretrained_model_2.most_similar(positive=["war"], topn=10)
print("Pretrained2:", result, "\n")

print("---------Query 4: human + history---------\n")

result = cbow_model.wv.most_similar(positive=["human", "history"], topn=10)
print("CBOW:", result, "\n")

result = skipgram_model.wv.most_similar(positive=["human", "history"], topn=10)
print("skipgram:", result, "\n")

result = pretrained_model_1.most_similar(positive=["human", "history"], topn=10)
print("Pretrained1:", result, "\n")

result = pretrained_model_2.most_similar(positive=["human", "history"], topn=10)
print("Pretrained2:", result, "\n")

print("---------Query 5: live + air---------\n")
result = cbow_model.wv.most_similar(positive=["live", "air"], topn=10)
print("CBOW:", result, "\n")

result = skipgram_model.wv.most_similar(positive=["live", "air"], topn=10)
print("skipgram:", result, "\n")

result = pretrained_model_1.most_similar(positive=["live", "air"], topn=10)
print("Pretrained1:", result, "\n")

result = pretrained_model_2.most_similar(positive=["live", "air"], topn=10)
print("Pretrained2:", result, "\n")

print("---------------------------------------------\n")