import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for
import random

app = Flask(__name__)

# Load and prepare data
def load_data():
    # Load your product data
    # This would be replaced with your actual data loading logic
    # For example: df = pd.read_csv('your_products_file.csv')
    
    # Sample data based on what you provided
    data = {
        'Name': ["Nicole by OPI Nail Lacquer, Next Stop the Bikini Zone A59, .5 fl oz", 
                "R+Co Aircraft Pomade Mousse, 5.6 Oz",
                "Candle Warmers Etc. Rustic Brown Hurricane Candle Warmer Lantern",
                "Recovery Complex Anti-Frizz Shine Serum by Bain de Terre for Unisex, 4.2 oz",
                "ReNew Life CleanseMore, Veggie Caps, 60 ea",
                "Alba Botanica Very Emollient Herbal Healing Body Lotion, 32 oz.",
                "Groganics DHT Ice Oil Scalp Moisturizer, 4 oz",
                "Vega Chlorella Dietary Supplement Powder 5.3 oz. Bottle",
                "Guerlain L'homme Ideal Eau De Toilette Spray for Men 3.3 oz",
                "Alaffia Body Lotion, Vanilla, 32 Oz"],
        'ReviewCount': [1, 1, 10, 4, 15, 3, 1, 1, 1, 2],
        'Brand': ['opi', '', 'candle, warmers, etc', 'bain, de, terre', 'renew, life', 
                 'alba, botanica', 'groganics', 'vega', 'guerlain', 'alaffia'],
        'ImageURL': [
            "https://i5.walmartimages.com/asr/a3436bdc-e2e5-4c0c-b55c-0b2cbfbd7757_1.dfbc7c5baecd7674a3dfb60c84daf4b7.jpeg",
            "https://i5.walmartimages.com/asr/03319cbe-7f61-42d3-afa9-4c2ac5e2342e.2b236bcbb74ce2f85e3d3160d9b52236.jpeg",
            "https://i5.walmartimages.com/asr/54376245-b5c1-4d6a-9972-bc41a2a825ea_1.f46b3671e8d222adc37867e197457837.png",
            "https://i5.walmartimages.com/asr/fcdb4d2e-3727-4bc4-bb2a-63c585c236b0_1.4c8c7111e5dde79bad7e54b6f71a8781.jpeg",
            "https://i5.walmartimages.com/asr/9f707fe4-9ee3-4dc5-b230-0005d2ba6f29_1.3b8ea51118f73b8528bbc6b808dd4ba4.jpeg",
            "https://i5.walmartimages.com/asr/6050a2f0-3f91-4fb5-a0d3-07878bbe0f21.2ba494455079455da1f24ff0a193245c.jpeg",
            "https://i5.walmartimages.com/asr/ed63df5f-b0a6-44d9-b38a-5385a3705609_1.40863f65eaff18db5473f244c56dc91e.jpeg",
            "https://i5.walmartimages.com/asr/e8ddd649-4959-4454-9798-cc185525baa6_1.c98f8adaf041556d63baadedf00316a9.jpeg",
            "https://i5.walmartimages.com/asr/2f141245-7503-494e-8354-17ae43e807b5_1.5490debb7c8f5c5ea1e03889242cd3da.jpeg",
            "https://i5.walmartimages.com/asr/2988c323-cb6f-4a45-9bd7-9029d981630c_1.d65b6410f1b5a72233cdab07e25e153b.jpeg"
        ],
        'Rating': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'ProductID': [2609, 3124, 885, 3140, 3137, 436, 1704, 3846, 1707, 428]
    }
    
    # Create dataframe
    products_df = pd.DataFrame(data)
    
    # For simplicity, let's create a simple user-item matrix with simulated ratings
    # In a real scenario, this would be your actual user-item interaction data
    
    # Create some simulated user ratings (this would be replaced with real user data)
    # For this example, we'll create 20 simulated users
    num_users = 20
    num_products = len(products_df)
    
    # Initialize a matrix with NaN values (representing no rating)
    user_item_matrix = np.full((num_users, num_products), np.nan)
    
    # Randomly assign some ratings (1-5) to simulate user behavior
    # Each user rates between 3 and 7 products
    np.random.seed(42)  # For reproducibility
    for user_id in range(num_users):
        # Decide how many products this user will rate
        num_ratings = np.random.randint(3, min(8, num_products))
        
        # Randomly choose which products to rate
        products_to_rate = np.random.choice(range(num_products), num_ratings, replace=False)
        
        # Assign ratings (mostly high ratings, with some variation)
        for prod_idx in products_to_rate:
            # Higher probability of high ratings (4-5) with some lower ratings
            rating_weights = [0.05, 0.1, 0.15, 0.3, 0.4]  # Weights for ratings 1-5
            rating = np.random.choice(range(1, 6), p=rating_weights)
            user_item_matrix[user_id, prod_idx] = rating
    
    # Convert to pandas DataFrame for easier manipulation
    user_ratings_df = pd.DataFrame(user_item_matrix, 
                                 columns=products_df['ProductID'],
                                 index=[f'User_{i}' for i in range(num_users)])
    
    return products_df, user_ratings_df

# Function to get similar users based on ratings
def get_similar_users(user_ratings_df, active_user_id, n=5):
    """
    Find similar users to the active user based on rating patterns
    
    Parameters:
    user_ratings_df (DataFrame): User-item ratings matrix
    active_user_id (int): ID of the active user
    n (int): Number of similar users to return
    
    Returns:
    list: Indices of similar users
    """
    # Convert user_id to dataframe index if needed
    if isinstance(active_user_id, str) and active_user_id.startswith('User_'):
        active_user_idx = int(active_user_id.split('_')[1])
    else:
        active_user_idx = active_user_id
    
    # Get the active user's ratings
    active_user_ratings = user_ratings_df.iloc[active_user_idx].values.reshape(1, -1)
    
    # Fill NaN values with 0 for calculation purposes
    active_user_ratings = np.nan_to_num(active_user_ratings, nan=0.0)
    ratings_matrix = user_ratings_df.fillna(0).values
    
    # Compute cosine similarity between the active user and all other users
    similarities = cosine_similarity(active_user_ratings, ratings_matrix)[0]
    
    # Get indices of most similar users (excluding the active user)
    similar_user_indices = similarities.argsort()[::-1][1:n+1]
    
    return similar_user_indices

# Function to get collaborative recommendations
def get_collaborative_recommendations(products_df, user_ratings_df, active_user_id, n=5):
    """
    Get collaborative filtering recommendations for a user
    
    Parameters:
    products_df (DataFrame): Product information
    user_ratings_df (DataFrame): User-item ratings matrix
    active_user_id (int): ID of the active user
    n (int): Number of recommendations to return
    
    Returns:
    DataFrame: Top n recommended products
    """
    # Get similar users
    similar_user_indices = get_similar_users(user_ratings_df, active_user_id)
    
    # Get the active user's ratings
    if isinstance(active_user_id, str) and active_user_id.startswith('User_'):
        active_user_idx = int(active_user_id.split('_')[1])
    else:
        active_user_idx = active_user_id
        
    active_user_ratings = user_ratings_df.iloc[active_user_idx]
    
    # Find products the active user hasn't rated yet
    unrated_products = active_user_ratings[active_user_ratings.isna()].index
    
    # If the user has rated all products, just return the top-rated products
    if len(unrated_products) == 0:
        # Get the user's top-rated products
        top_rated = active_user_ratings.sort_values(ascending=False).index[:n]
        return products_df[products_df['ProductID'].isin(top_rated)]
    
    # Calculate predicted ratings for unrated products
    predicted_ratings = {}
    
    for product_id in unrated_products:
        # Get ratings for this product from similar users
        similar_users_ratings = []
        
        for idx in similar_user_indices:
            rating = user_ratings_df.iloc[idx][product_id]
            if not np.isnan(rating):
                similar_users_ratings.append(rating)
        
        # If at least one similar user rated this product, calculate the average rating
        if similar_users_ratings:
            predicted_ratings[product_id] = np.mean(similar_users_ratings)
    
    # Sort predictions by predicted rating (descending)
    sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top n product IDs
    top_product_ids = [product_id for product_id, _ in sorted_predictions[:n]]
    
    # Return the product information for the recommended products
    recommended_products = products_df[products_df['ProductID'].isin(top_product_ids)]
    
    return recommended_products

# Function to get content-based recommendations based on search query
def get_content_based_recommendations(products_df, search_query, n=5):
    """
    Get content-based recommendations based on search query
    
    Parameters:
    products_df (DataFrame): Product information
    search_query (str): Search term from the user
    n (int): Number of recommendations to return
    
    Returns:
    DataFrame: Top n recommended products matching the search query
    """
    # Convert search query to lowercase for case-insensitive matching
    search_query = search_query.lower()
    
    # Look for matches in product name and brand
    products_df['Name_lower'] = products_df['Name'].str.lower()
    products_df['Brand_lower'] = products_df['Brand'].str.lower()
    
    # Find matches in name or brand
    name_matches = products_df[products_df['Name_lower'].str.contains(search_query, na=False)]
    brand_matches = products_df[products_df['Brand_lower'].str.contains(search_query, na=False)]
    
    # Combine matches
    matches = pd.concat([name_matches, brand_matches]).drop_duplicates()
    
    # Clean up temporary columns
    matches = matches.drop(['Name_lower', 'Brand_lower'], axis=1)
    
    # If no matches were found, return an empty DataFrame
    if len(matches) == 0:
        return pd.DataFrame(columns=products_df.columns)
    
    # Return top n matches (based on review count or other relevance metrics)
    matches = matches.sort_values(by='ReviewCount', ascending=False).head(n)
    
    return matches

# Function to get trending products
def get_trending_products(products_df, n=4):
    """
    Get trending products based on review count and rating
    
    Parameters:
    products_df (DataFrame): Product information
    n (int): Number of trending products to return
    
    Returns:
    DataFrame: Top n trending products
    """
    # Sort products by a combination of review count and rating
    # This is a simple way to get "trending" products
    products_df['trending_score'] = products_df['ReviewCount'] * products_df['Rating']
    trending = products_df.sort_values(by='trending_score', ascending=False).head(n)
    trending = trending.drop('trending_score', axis=1)
    
    return trending

# Generate random product image URLs for the homepage
def generate_random_product_images(n=4):
    """Generate random product image URLs for the homepage"""
    # List of sample image URLs (could be expanded or replaced with actual data)
    sample_urls = [
        "https://i5.walmartimages.com/asr/a3436bdc-e2e5-4c0c-b55c-0b2cbfbd7757_1.dfbc7c5baecd7674a3dfb60c84daf4b7.jpeg",
        "https://i5.walmartimages.com/asr/03319cbe-7f61-42d3-afa9-4c2ac5e2342e.2b236bcbb74ce2f85e3d3160d9b52236.jpeg",
        "https://i5.walmartimages.com/asr/54376245-b5c1-4d6a-9972-bc41a2a825ea_1.f46b3671e8d222adc37867e197457837.png",
        "https://i5.walmartimages.com/asr/fcdb4d2e-3727-4bc4-bb2a-63c585c236b0_1.4c8c7111e5dde79bad7e54b6f71a8781.jpeg",
        "https://i5.walmartimages.com/asr/9f707fe4-9ee3-4dc5-b230-0005d2ba6f29_1.3b8ea51118f73b8528bbc6b808dd4ba4.jpeg",
        "https://i5.walmartimages.com/asr/6050a2f0-3f91-4fb5-a0d3-07878bbe0f21.2ba494455079455da1f24ff0a193245c.jpeg",
        "https://i5.walmartimages.com/asr/ed63df5f-b0a6-44d9-b38a-5385a3705609_1.40863f65eaff18db5473f244c56dc91e.jpeg",
        "https://i5.walmartimages.com/asr/e8ddd649-4959-4454-9798-cc185525baa6_1.c98f8adaf041556d63baadedf00316a9.jpeg",
    ]
    
    # Select random URLs from the sample list
    random_urls = random.sample(sample_urls, min(n, len(sample_urls)))
    
    # If we need more URLs than in our sample list, duplicate some
    while len(random_urls) < n:
        random_urls.append(random.choice(sample_urls))
    
    return random_urls



# Main route for index.html
@app.route('/')
def index():
    products_df, _ = load_data()
    
    # Get trending products for the homepage
    trending_products = get_trending_products(products_df, n=4)
    
   
    
    # Generate random image URLs for the trending products
    random_product_image_urls = generate_random_product_images(len(trending_products))
    
    # Generate a random price for demo purposes
    random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
    
    return render_template('index.html', 
                         trending_products=trending_products,
                         random_product_image_urls=random_product_image_urls,
                         random_price=random_price)

# Route for the main page
@app.route('/main.html')
def main():
    # Create an empty DataFrame with the right columns for the initial main page load
    content_based_rec = pd.DataFrame(columns=['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'ProductID'])
    # Pass a random price for the template
    random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
    # No message initially
    message = ""
    return render_template('main.html', content_based_rec=content_based_rec, random_price=random_price, message=message)

# Route for recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get search query and number of recommendations from form
    search_query = request.form.get('prod', '')
    try:
        num_recommendations = int(request.form.get('nbr', 5))
    except ValueError:
        num_recommendations = 5
    
    # Load product data
    products_df, user_ratings_df = load_data()
    
    # First, try to find products that match the search query
    content_based_rec = get_content_based_recommendations(products_df, search_query, n=num_recommendations)
    
    # If no matches are found for the search query, return "no recommendation" message
    if len(content_based_rec) == 0:
        # Create an empty DataFrame with the right columns
        empty_df = pd.DataFrame(columns=products_df.columns)
        message = f"No recommendation found for '{search_query}'"
        random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
        return render_template('main.html', 
                            content_based_rec=empty_df, 
                            message=message,
                            random_price=random_price)
    
    # If direct matches found, limit to exactly the requested number (or available matches)
    content_based_rec = content_based_rec.head(num_recommendations)
    
    # Generate a random price for each product (for demonstration)
    random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
    
    message = f"Showing {len(content_based_rec)} products related to '{search_query}'"
    
    return render_template('main.html', 
                        content_based_rec=content_based_rec, 
                        message=message,
                        random_price=random_price)

# Truncate function for Jinja template (to limit text length)
def truncate(text, length):
    if len(text) <= length:
        return text
    return text[:length] + "..."

# Add the function to Jinja environment
app.jinja_env.globals.update(truncate=truncate)

# Routes for signup and signin
@app.route('/signup', methods=['POST'])
def signup():
    # Get form data
    username = request.form.get('username', '')
    email = request.form.get('email', '')
    password = request.form.get('password', '')
    
    # In a real system, you would add user data to a database
    # For now, just return to index with a success message
    
    # Load product data for trending section
    products_df, _ = load_data()
    trending_products = get_trending_products(products_df, n=4)
    random_product_image_urls = generate_random_product_images(len(trending_products))
    random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
    
    return render_template('index.html',
                         trending_products=trending_products,
                         random_product_image_urls=random_product_image_urls,
                         random_price=random_price,
                         signup_message=f"Welcome {username}! Your account has been created.")

@app.route('/signin', methods=['POST'])
def signin():
    # Get form data
    username = request.form.get('signinUsername', '')
    password = request.form.get('signinPassword', '')
    
    # In a real system, you would verify user credentials
    # For now, just return to index with a success message
    
    # Load product data for trending section
    products_df, _ = load_data()
    trending_products = get_trending_products(products_df, n=4)
    random_product_image_urls = generate_random_product_images(len(trending_products))
    random_price = f"${random.randint(10, 100)}.{random.randint(0, 99):02d}"
    
    return render_template('index.html',
                         trending_products=trending_products,
                         random_product_image_urls=random_product_image_urls,
                         random_price=random_price,
                         signup_message=f"Welcome back, {username}!")

if __name__ == '__main__':
    app.run(debug=True)