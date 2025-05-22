import csv
import time

from services.firebase_config import db
from services.user_id_mapper import get_numeric_user_id
import pandas as pd

from utils.trigger_generate_user import trigger_generate_user

def generate_ratings_csv(firebase_uid):
    user_id = get_numeric_user_id(firebase_uid)
    new_ratings = []

    print("🔍 Starting Firestore query for user reviews...")
    user_reviews = db.collection("reviews") \
        .where("user_id", "==", firebase_uid) \
        .stream()
    print("✅ Firestore query finished.")

    for review in user_reviews:
        review_data = review.to_dict()
        try:
            rating_value = float(review_data.get('overallRating', 5))
        except (TypeError, ValueError):
            rating_value = 5  # fallback

        location_id = review_data.get('locationId')
        if not location_id:
            continue

        new_ratings.append({
            'userId': user_id,
            'movieId': location_id,
            'rating': rating_value,
            'timestamp': int(time.time())
        })

    file_path = 'ml-latest-small/ratings.csv'

    try:
        existing_df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

    filtered_df = existing_df[existing_df['userId'] != user_id]
    new_df = pd.DataFrame(new_ratings)
    combined_df = pd.concat([filtered_df, new_df], ignore_index=True)
    combined_df.to_csv(file_path, index=False)

    print(f'✅ ratings.csv updated for Firebase UID {firebase_uid} (user ID {user_id})')

# def generate_ratings_csv(firebase_uid):
#     user_id = get_numeric_user_id(firebase_uid)
#     new_ratings = []

#     locations_ref = db.collection('locations')
#     location_docs = locations_ref.stream()

#     for location in location_docs:
#         reviews_ref = location.reference.collection('reviews')
#         #user_reviews = reviews_ref.where('userId', '==', firebase_uid).stream()
#         print("🔍 Starting Firestore query for user reviews...")
#         user_reviews = db.collection("reviews") \
#             .where("user_id", "==", firebase_uid) \
#             .limit(50).stream()
#         print("✅ Firestore query finished.")


#         for review in user_reviews:
#             review_data = review.to_dict()
#             try:
#                 rating_value = float(review_data.get('overallRating', 5))
#             except (TypeError, ValueError):
#                 rating_value = 5  # fallback if something's wrong

#             new_ratings.append({
#                 'userId': user_id,
#                 'movieId': location.id,
#                 'rating': rating_value,
#                 'timestamp': int(time.time())
#             })



#     file_path = 'ml-latest-small/ratings.csv'

#     try:
#         existing_df = pd.read_csv(file_path, encoding='ISO-8859-1')
#     except FileNotFoundError:
#         existing_df = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

#     # Remove existing ratings from same user
#     filtered_df = existing_df[existing_df['userId'] != user_id]
#     new_df = pd.DataFrame(new_ratings)
#     combined_df = pd.concat([filtered_df, new_df], ignore_index=True)
#     combined_df.to_csv(file_path, index=False)

#     print(f'✅ ratings.csv updated for Firebase UID {firebase_uid} (user ID {user_id})')
