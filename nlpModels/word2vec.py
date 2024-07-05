import gensim
from gensim.models import Word2Vec


def train_word2vec(corpus):
    tokenized_corpus = [sentence.lower().split() for sentence in corpus]
    word2vec_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=10)
    word2vec_model.save("word2vec_model.bin")
    return word2vec_model

def load_word2vec_model(model_path):
    # Load Word2Vec model
    return Word2Vec.load(model_path)