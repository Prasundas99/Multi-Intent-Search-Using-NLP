import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from data.searchData import restaurants_data

def train_lstm(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    word_index = tokenizer.word_index
    max_sequence_length = max(len(seq) for seq in sequences)
    data = pad_sequences(sequences, maxlen=max_sequence_length)
    labels = np.random.randint(2, size=(len(data), 1))
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(data, labels, epochs=10, batch_size=4)
    return model, tokenizer, max_sequence_length

def encode_query(query, tokenizer, max_sequence_length):
    seq = tokenizer.texts_to_sequences([query])
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length)
    return padded_seq

def nlpSearch(query, lstm_model, word2vec_model, tokenizer, max_sequence_length):
    # Find most similar restaurant based on the query using Word2Vec
    tokenized_query = query.lower().split()  # Ensure query is lowercase
    restaurant_similarities = []
    for restaurant in restaurants_data:
        restaurant_name_tokens = restaurant['name'].lower().split()  # Tokenize restaurant name
       # Calculate similarity for each word in the query and average them
        similarity_scores = [word2vec_model.wv.similarity(word, restaurant_token) 
                             for word in tokenized_query 
                             for restaurant_token in restaurant_name_tokens # Iterate over tokens in restaurant name
                             if word in word2vec_model.wv and restaurant_token in word2vec_model.wv] # Check if both words are in vocabulary
        if similarity_scores:  # Check if any similarity scores were found
            average_similarity = sum(similarity_scores) / len(similarity_scores)
            restaurant_similarities.append((restaurant['name'].lower(), average_similarity)) # Append the lowercased restaurant name

    restaurant_similarities.sort(key=lambda x: x[1], reverse=True)
    if restaurant_similarities:  # Handle the case where no similar restaurants are found
        top_restaurant_name = restaurant_similarities[0][0]
    else:
        return "No similar restaurants found."

    # ... rest of the function remains the same

    # Encode query for LSTM
    encoded_query = encode_query(query, tokenizer, max_sequence_length)

    # Perform LSTM prediction
    prediction = lstm_model.predict(encoded_query)

    # Find the restaurant and menu item details
    restaurant_result = next((r for r in restaurants_data if r['name'].lower() == top_restaurant_name), None)

    if restaurant_result:
        return {
            "restaurant": restaurant_result['name'],
            "menu": restaurant_result['menu'],
            "prediction": prediction
        }
    else:
        return "No matching restaurant found."    

