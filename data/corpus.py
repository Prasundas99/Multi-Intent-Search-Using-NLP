from data.searchData import restaurants_data

corpus = [
    "Margherita Pizza", "Pepperoni Pizza", "Veggie Pizza",
    "Classic Burger", "Cheese Burger", "Veggie Burger",
    "California Roll", "Spicy Tuna Roll", "Avocado Roll",
    "Spaghetti Bolognese", "Fettuccine Alfredo", "Penne Arrabbiata",
    "Butter Chicken", "Paneer Tikka Masala", "Dal Makhani",
    "Chicken Taco", "Beef Taco", "Veggie Taco",
    "Caesar Salad", "Greek Salad", "Garden Salad",
    "BBQ Chicken Wings", "BBQ Ribs", "Grilled Veggies",
    "Classic Pancakes", "Blueberry Pancakes", "Chocolate Chip Pancakes",
    "Strawberry Smoothie", "Mango Smoothie", "Green Smoothie"
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