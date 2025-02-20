from wefe.metrics import WEAT
from wefe.query import Query
from gensim.models import Word2Vec, KeyedVectors
from wefe.word_embedding_model import WordEmbeddingModel  # Required to wrap embeddings

lost_words_threshold = 0.7  # You can set this to any value between 0 and 1

# load pre-trained models
cbow_model = Word2Vec.load("cbow_wikipedia.model").wv
skipgram_model = Word2Vec.load("skipgram_wikipedia.model").wv
pretrained_model_1 = KeyedVectors.load_word2vec_format('./model11.bin', binary=True)
pretrained_model_2 = KeyedVectors.load_word2vec_format('./model12.bin', binary=True)

# Wrap the models in WEFE's WordEmbeddingModel
cbow_wefe = WordEmbeddingModel(cbow_model, name="CBOW")
skipgram_wefe = WordEmbeddingModel(skipgram_model, name="Skip-gram")
pretrained_wefe_1 = WordEmbeddingModel(pretrained_model_1, name="Pretrained 1")
pretrained_wefe_2 = WordEmbeddingModel(pretrained_model_2, name="Pretrained 2")


# define queries for WEAT 
queries = [
    Query(
        target_sets=[["man", "doctor"], ["woman", "nurse"]],
        attribute_sets=[["career", "business"], ["family", "home"]],
        target_sets_names=["Male-associated", "Female-associated"],
        attribute_sets_names=["Career", "Family"],
    ),
]

weat = WEAT()

# List of models
models = [cbow_wefe, skipgram_wefe, pretrained_wefe_1, pretrained_wefe_2]

# Run WEAT on multiple models
for model in models:
    print(f"Results for {model}:")
    for query in queries:
        try:
            result = weat.run_query(query, model)
            effect_size = result.get("effect_size", "N/A") 
            print(f"Query: {query.query_name}, Effect Size: {effect_size}")
        except Exception as e:
            print(f"Error with {model.model_name}: {e}")
    print("-" * 50)