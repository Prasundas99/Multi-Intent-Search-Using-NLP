# Multi Intent Search Using NLP
This project demonstrates a multi-intent search system for restaurant data using natural language processing (NLP). The system leverages Byte Pair Encoding (BPE) for efficient tokenization, Word2Vec for semantic similarity, and an LSTM model for sequence prediction. The primary goal is to match user queries with relevant restaurant names and menu items.

## Data Architecture

![image](https://github.com/Prasundas99/Multi-Intent-Search-Using-NLP/assets/58937669/88c067c7-f33e-4ccf-a765-796835127c1b)

###  Dataset
The dataset comprises restaurant information, including restaurant names, menu items, prices, and distances. Each restaurant has a unique ID and a list of menu items with their respective details.

### Corpus
A corpus is generated from the dataset, which includes all restaurant names and menu items. This corpus is used for training the models. The names and menu items are converted to lowercase to ensure consistency during training and querying.

### Models
Byte Pair Encoding (BPE)
Byte Pair Encoding is a subword tokenization technique that helps efficiently handle rare or out-of-vocabulary words by breaking them into more common subwords. BPE is used to preprocess the corpus before training the models, ensuring better tokenization and improved model performance.

### Word2Vec
Word2Vec is used to capture semantic similarities between words. It helps in finding the most similar restaurant based on user queries. By training on the generated corpus, the model learns vector representations of words, allowing it to compute similarities between user queries and restaurant names or menu items.

### LSTM
An LSTM (Long Short-Term Memory) model is used for sequence prediction, helping to capture the context of user queries. The LSTM model is trained on sequences generated from the corpus, allowing it to understand the relationships and patterns within the text data.

### Query Processing
The system processes user queries by leveraging the BPE, Word2Vec, and LSTM models to find the most relevant restaurant and menu items. The BPE tokenizer breaks down the query into subwords, which are then used by the Word2Vec model to compute similarity with the restaurant names or menu items. The LSTM model

### Usage
- Generate Corpus: Extract restaurant names and menu items from the dataset to create a training corpus.
- Preprocess with BPE: Use Byte Pair Encoding to preprocess the corpus for better tokenization.
- Train Word2Vec Model: Train the Word2Vec model on the preprocessed corpus to learn word embeddings.
- Train LSTM Model: Train the LSTM model on sequences from the corpus to understand text patterns and relationships.
- Query Processing: Use the trained models to process user queries and return the most relevant restaurant and menu items.

### Note
If main.py is not working as expected, please refer to collab.py for the complete project implementation.

### Conclusion
This project illustrates a multi-intent NLP search system for restaurants using Word2Vec and LSTM, enhanced with Byte Pair Encoding for subword tokenization. By integrating these techniques, the system can efficiently match user queries with relevant restaurant names and menu items based on semantic similarity and sequence prediction, providing accurate and contextually relevant search results.
