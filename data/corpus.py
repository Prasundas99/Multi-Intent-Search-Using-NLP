from data.searchData import restaurants_data

corpus = []

def generateCorpus():
    for restaurant in restaurants_data:
        # Extract restaurant details
        restaurant_name = restaurant['name']
        menu_items = restaurant['menu']
        distance = restaurant['distance_m']
        
        # Create sentences for each restaurant with more details
        restaurant_sentences = []
        
        # Sentence 1: Restaurant name
        name_sentence = f"The restaurant's name is {restaurant_name}."
        restaurant_sentences.append(name_sentence)
        
        # Sentence 2: Menu items with prices
        menu_sentences = []
        for item in menu_items:
            item_name = item['name']
            item_price = item['price']
            item_sentence = f"They serve {item_name} for {item_price} rupees."
            menu_sentences.append(item_sentence)
        restaurant_sentences.extend(menu_sentences)
        
        # Sentence 3: Distance from a location
        distance_description = f"It is located {distance} meters away."
        restaurant_sentences.append(distance_description)
        
        # Sentence 4: Example of a menu item
        if len(menu_items) > 0:
            example_item = menu_items[0]['name']
            example_sentence = f"For example, they serve {example_item}."
            restaurant_sentences.append(example_sentence)
        
        # Combine all sentences for the restaurant into one paragraph
        restaurant_paragraph = ' '.join(restaurant_sentences)
        
        # Append the paragraph to the corpus
        corpus.append(restaurant_paragraph)



