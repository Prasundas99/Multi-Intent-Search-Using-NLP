# !pip install fastapi
# !pip install uvicorn
# !pip install gensim
# !pip install tensorflow
# !pip install pydantic
# !pip install numpy
# !pip install scikit-learn
# !pip install python-dotenv
# !pip install tensorflow-addons
# !pip install torch


restaurants_data = [
    {
        "restaurant_id": 1,
        "name": "Pizza Palace",
        "menu": [
            {"item_id": 101, "name": "Margherita Pizza", "price": 250},
            {"item_id": 102, "name": "Pepperoni Pizza", "price": 300},
            {"item_id": 103, "name": "Veggie Pizza", "price": 280}
        ],
        "distance_m": 500
    },
    {
        "restaurant_id": 2,
        "name": "Burger Bonanza",
        "menu": [
            {"item_id": 201, "name": "Classic Burger", "price": 150},
            {"item_id": 202, "name": "Cheese Burger", "price": 180},
            {"item_id": 203, "name": "Veggie Burger", "price": 160}
        ],
        "distance_m": 800
    },
    {
        "restaurant_id": 3,
        "name": "Sushi Central",
        "menu": [
            {"item_id": 301, "name": "California Roll", "price": 350},
            {"item_id": 302, "name": "Spicy Tuna Roll", "price": 400},
            {"item_id": 303, "name": "Avocado Roll", "price": 300}
        ],
        "distance_m": 1200
    },
    {
        "restaurant_id": 4,
        "name": "Pasta Point",
        "menu": [
            {"item_id": 401, "name": "Spaghetti Bolognese", "price": 320},
            {"item_id": 402, "name": "Fettuccine Alfredo", "price": 350},
            {"item_id": 403, "name": "Penne Arrabbiata", "price": 300}
        ],
        "distance_m": 200
    },
    {
        "restaurant_id": 5,
        "name": "Curry Corner",
        "menu": [
            {"item_id": 501, "name": "Butter Chicken", "price": 280},
            {"item_id": 502, "name": "Paneer Tikka Masala", "price": 260},
            {"item_id": 503, "name": "Dal Makhani", "price": 240}
        ],
        "distance_m": 600
    },
    {
        "restaurant_id": 6,
        "name": "Taco Town",
        "menu": [
            {"item_id": 601, "name": "Chicken Taco", "price": 120},
            {"item_id": 602, "name": "Beef Taco", "price": 140},
            {"item_id": 603, "name": "Veggie Taco", "price": 100}
        ],
        "distance_m": 900
    },
    {
        "restaurant_id": 7,
        "name": "Salad Stop",
        "menu": [
            {"item_id": 701, "name": "Caesar Salad", "price": 200},
            {"item_id": 702, "name": "Greek Salad", "price": 180},
            {"item_id": 703, "name": "Garden Salad", "price": 150}
        ],
        "distance_m": 300
    },
    {
        "restaurant_id": 8,
        "name": "BBQ Bliss",
        "menu": [
            {"item_id": 801, "name": "BBQ Chicken Wings", "price": 350},
            {"item_id": 802, "name": "BBQ Ribs", "price": 500},
            {"item_id": 803, "name": "Grilled Veggies", "price": 250}
        ],
        "distance_m": 1500
    },
    {
        "restaurant_id": 9,
        "name": "Pancake House",
        "menu": [
            {"item_id": 901, "name": "Classic Pancakes", "price": 150},
            {"item_id": 902, "name": "Blueberry Pancakes", "price": 200},
            {"item_id": 903, "name": "Chocolate Chip Pancakes", "price": 220}
        ],
        "distance_m": 700
    },
    {
        "restaurant_id": 10,
        "name": "Smoothie Shack",
        "menu": [
            {"item_id": 1001, "name": "Strawberry Smoothie", "price": 150},
            {"item_id": 1002, "name": "Mango Smoothie", "price": 160},
            {"item_id": 1003, "name": "Green Smoothie", "price": 180}
        ],
        "distance_m": 400
    },
    {
        "restaurant_id": 11,
        "name": "KFC",
        "menu": [
            {"item_id": 1101, "name": "Original Recipe Chicken", "price": 300},
            {"item_id": 1102, "name": "Zinger Burger", "price": 200},
            {"item_id": 1103, "name": "Hot Wings", "price": 180}
        ],
        "distance_m": 1000
    },
    {
        "restaurant_id": 12,
        "name": "Domino's",
        "menu": [
            {"item_id": 1201, "name": "Pepperoni Pizza", "price": 280},
            {"item_id": 1202, "name": "Cheese Burst Pizza", "price": 300},
            {"item_id": 1203, "name": "Chicken Dominator", "price": 350}
        ],
        "distance_m": 1100
    },
    {
        "restaurant_id": 13,
        "name": "McDonald's",
        "menu": [
            {"item_id": 1301, "name": "Big Mac", "price": 250},
            {"item_id": 1302, "name": "McChicken", "price": 220},
            {"item_id": 1303, "name": "McVeggie", "price": 200}
        ],
        "distance_m": 500
    },
    {
        "restaurant_id": 14,
        "name": "Arsenal",
        "menu": [
            {"item_id": 1401, "name": "Classic Fish and Chips", "price": 320},
            {"item_id": 1402, "name": "Shepherd's Pie", "price": 350},
            {"item_id": 1403, "name": "Beef Wellington", "price": 500}
        ],
        "distance_m": 1300
    },
    {
        "restaurant_id": 15,
        "name": "Amnesia",
        "menu": [
            {"item_id": 1501, "name": "Grilled Salmon", "price": 400},
            {"item_id": 1502, "name": "Steak Frites", "price": 450},
            {"item_id": 1503, "name": "Lobster Bisque", "price": 350}
        ],
        "distance_m": 1400
    }
]


corpus = [
    "Margherita Pizza", "Pepperoni Pizza", "Veggie Pizza",
    "Classic Burger", "Cheese Burger", "Veggie Burger",
    "California Roll", "Spicy Tuna Roll", "Avocado Roll",
    "Spaghetti Bolognese", "Fettuccine Alfredo", "Penne Arrabbiata",
    "Butter Chicken", "Paneer Tikka Masala", "Dal Makhani",
    "Chicken Taco", "Beef Taco", "Veggie Taco",
    "Caesar Salad", "Greek Salad", "Garden Salad", "BBQ chicken",
    "BBQ Chicken Wings", "BBQ Ribs", "Grilled Veggies",
    "Classic Pancakes", "Blueberry Pancakes", "Chocolate Chip Pancakes",
    "Strawberry Smoothie", "Mango Smoothie", "Green Smoothie", "pizza palace",
    "KFC", "Domino's", "McDonald's", "Arsenal", "Amnesia"
]

def generateCorpus():
    corpus = []
    for restaurant in restaurants_data:
        corpus.append(restaurant['name'].lower())
        corpus.extend(restaurant['name'].lower().split())
        for menu_item in restaurant['menu']:
            corpus.append(menu_item['name'].lower())
            corpus.extend(menu_item['name'].lower().split())
    return corpus


import gensim
from gensim.models import Word2Vec


def train_word2vec(corpus):
    # Tokenize corpus
    tokenized_corpus = [sentence.lower().split() for sentence in corpus]

    # Train Word2Vec model
    word2vec_model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=10)
    # Save Word2Vec model
    word2vec_model.save("word2vec_model.bin")
    return word2vec_model

def load_word2vec_model(model_path):
    # Load Word2Vec model
    return Word2Vec.load(model_path)   



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def trainLstm(corpus):
    # Prepare text data
    all_texts = corpus  # Combine restaurant names and menu items into a single list
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    sequences = tokenizer.texts_to_sequences(all_texts)
    word_index = tokenizer.word_index

    # Pad sequences
    max_sequence_length = max(len(seq) for seq in sequences)
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Prepare labels (for simplicity, let's assume we have binary relevance labels)
    # In a real-world scenario, these labels would come from user interactions or ratings
    labels = np.random.randint(2, size=(len(data), 1))

    # Create the LSTM model
    embedding_dim = 100
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    model.fit(data, labels, epochs=10, batch_size=4)
    
    # return model tokenizer and max_sequence_length
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


query = "BBQ chicken"

print("Generating corpus")
corpusData = generateCorpus()
print("Corpus generated", corpusData )

print("Training word2vec model")
w2vModel = train_word2vec(corpusData)
print("Word2vec model trained", w2vModel)

print("Training LSTM")
model, tokenizer, max_sequence_length = trainLstm(corpusData)
print("LSTM model trained")

print("All models trained")
results = nlpSearch(query, model,w2vModel, tokenizer, max_sequence_length)

print(f"Results for '{query}':")
print(results) 
