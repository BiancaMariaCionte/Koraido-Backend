
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import heapq
from collections import Counter, defaultdict
from operator import itemgetter
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic
import csv
from ai.ContentKNNAlgorithm import ContentKNNAlgorithm
from ai.Evaluator import Evaluator
from utils.generate_user_info import generate_user_info_xlsx
from utils.generate_ratings import generate_ratings_csv
from services.user_id_mapper import get_numeric_user_id
from services.firebase_config import db
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load Dataset and Train Models
# class MovieLens:
#     def __init__(self):
#         self.movieID_to_name = {}
#         self.name_to_movieID = {}

#         # Get path relative to current file (ai/app.py)
#         base_path = os.path.dirname(os.path.abspath(__file__))

#         self.ratingsPath = os.path.join(base_path, "ml-latest-small", "ratings.csv")
#         self.moviesPath = os.path.join(base_path, "ml-latest-small", "movies.csv")
#         self.userInfoPath = os.path.join(base_path, "ml-latest-small", "user_info.xlsx")

#         # Optionally, load them here if needed
#         ratings_df = pd.read_csv(self.ratingsPath)
#         movies_df = pd.read_csv(self.moviesPath, encoding="ISO-8859-1")
#         user_info_df = pd.read_excel(self.userInfoPath)
        
#         # Load ratings.csv
#        # ratings_path = os.path.join(base_path, "ml-latest-small", "ratings.csv")
#        # ratings_df = pd.read_csv(ratings_path)

#         # Load movies.csv
#        # movies_path = os.path.join(base_path, "ml-latest-small", "movies.csv")
#         #movies_df = pd.read_csv(movies_path, encoding="ISO-8859-1")

#         # Load user_info.xlsx
#        # user_info_path = os.path.join(base_path, "ml-latest-small", "user_info.xlsx")
#        # user_info_df = pd.read_excel(user_info_path)

#     def loadMovieLensLatestSmall(self):
#         reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
#         ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

#         with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
#             movieReader = csv.reader(csvfile)
#             next(movieReader)  # Skip header line
#             for row in movieReader:
#                 movieID = row[0]
#                 movieName = row[1]
#                 self.movieID_to_name[movieID] = movieName
#                 self.name_to_movieID[movieName] = movieID

#         return ratingsDataset

#     def getMovieName(self, movieID):
#         return self.movieID_to_name.get(movieID, "")
    
#     def loadUserInterests(self):
#         """Ensure user interests match genre IDs by using the same dictionary."""
#         user_interests = defaultdict(list)

#         # Ensure genres are already processed
#         if not hasattr(self, 'genreIDs'):
#             self.getGenres()  # Generate genre ID mappings if not already loaded

#         df = pd.read_excel(self.userInfoPath)

#         for _, row in df.iterrows():
#             userID = int(row['userId'])
#             interestList = row['interests'].split('|') if pd.notna(row['interests']) else []
#             interestIDList = []
        
#             for interest in interestList:
#                 if interest in self.genreIDs:  # Ensure consistency
#                     interestIDList.append(self.genreIDs[interest])  # Use genre ID mapping

#             user_interests[userID] = interestIDList  # Assign matched genre IDs to user interests
        
        
#         return user_interests
    
#     def getGenres(self):
#         genres = defaultdict(list)
#         self.genreIDs = {}  # Store genre IDs globally
#         maxGenreID = 0

#         with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
#             movieReader = csv.reader(csvfile)
#             next(movieReader)  # Skip header line
#             for row in movieReader:
#                 movieID = row[0]  # Keep as string
#                 genreList = row[2].split('|')  # Assuming genres are in the 3rd column
#                 genreIDList = []
#                 for genre in genreList:
#                     if genre in self.genreIDs:
#                         genreID = self.genreIDs[genre]
#                     else:
#                         genreID = maxGenreID
#                         self.genreIDs[genre] = genreID
#                         maxGenreID += 1
#                     genreIDList.append(genreID)
#                 genres[movieID] = genreIDList
#         return genres


# Your content-based KNN Algorithm class here, assumed imported or defined already
# from your_content_knn_module import ContentKNNAlgorithm

# Initialize Firebase Admin (only once, outside the class)


class MovieLens:
        def __init__(self):
                self.movieID_to_name = {}
                self.name_to_movieID = {}
                self.genreIDs = {}
                self.genres = defaultdict(list)  # movieID -> genreIDs

        def loadMoviesFromFirestore(self):
                movies_ref = db.collection('movies').stream()
                for doc in movies_ref:
                    movie = doc.to_dict()
                    movieID = str(movie['movieId'])
                    movieName = movie['title']
                    genreList = movie['genres'].split('|')
        
                    self.movieID_to_name[movieID] = movieName
                    self.name_to_movieID[movieName] = movieID
        
                    genreIDList = []
                    for genre in genreList:
                        if genre not in self.genreIDs:
                            self.genreIDs[genre] = len(self.genreIDs)
                        genreIDList.append(self.genreIDs[genre])
                    self.genres[movieID] = genreIDList

        def loadRatingsFromFirestore(self):
                ratings_ref = db.collection('ratings').stream()
                rating_data = []
                for doc in ratings_ref:
                    entry = doc.to_dict()
                    rating_data.append([str(entry['userId']), str(entry['movieId']), float(entry['rating'])])
                return rating_data

        def loadUserInterestsFromFirestore(self):
                users_ref = db.collection('user_info').stream()
                user_interests = defaultdict(list)
        
                for doc in users_ref:
                    data = doc.to_dict()
                    user_id = str(data.get('userId', doc.id))
                    interests = data.get('interests', "").split('|')
                    interestIDs = [self.genreIDs[genre] for genre in interests if genre in self.genreIDs]
                    user_interests[user_id] = interestIDs
        
                return user_interests

        def loadMovieLensLatestSmall(self):
                """Main method to initialize the dataset."""
                self.loadMoviesFromFirestore()
                ratings = self.loadRatingsFromFirestore()
                df_ratings = pd.DataFrame(ratings, columns=['userId', 'movieId', 'rating'])
        
                reader = Reader(rating_scale=(1, 5))
                return Dataset.load_from_df(df_ratings, reader)

        def getGenres(self):
                return self.genres

        def getMovieName(self, movieID):
                return self.movieID_to_name.get(movieID, "")

        def getMovieID(self, movieName):
                return self.name_to_movieID.get(movieName, "")

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()
trainSet = data.build_full_trainset()

# Train User-Based KNN Model
sim_options_user = {'name': 'cosine', 'user_based': True}
userKNN = KNNBasic(sim_options=sim_options_user)
userKNN.fit(trainSet)
user_sims = userKNN.compute_similarities()

# Train Item-Based KNN Model
sim_options_item = {'name': 'pearson', 'user_based': False}
itemKNN = KNNBasic(sim_options=sim_options_item)
itemKNN.fit(trainSet)
item_sims = itemKNN.compute_similarities()

# Train Content-Based KNN Model
contentKNN = ContentKNNAlgorithm()
contentKNN.fit(trainSet)



def get_user_based_recommendations(testUserInnerID):
    """Get User-Based Recommendations"""
    k = 10
    similarityRow = user_sims[testUserInnerID]
    similarUsers = [(innerID, score) for innerID, score in enumerate(similarityRow) if innerID != testUserInnerID]
    kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

    candidates = defaultdict(float)
    for similarUser in kNeighbors:
        innerID = similarUser[0]
        userSimilarityScore = similarUser[1]
        theirRatings = trainSet.ur[innerID]
        for rating in theirRatings:
            candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

    watched = {itemID: 1 for itemID, _ in trainSet.ur[testUserInnerID]}
    recommendations = []

    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if itemID not in watched:
            movieID = trainSet.to_raw_iid(itemID)
            recommendations.append({
                "id": movieID,
                "name": ml.getMovieName(movieID),
                "score": round(ratingSum, 3)
            })
        if len(recommendations) >= 10:
            break

    return recommendations

def get_item_based_recommendations(testUserInnerID):
    """Get Item-Based Recommendations"""
    k = 10
    testUserRatings = trainSet.ur[testUserInnerID]
    kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])

    candidates = defaultdict(float)
    for itemID, rating in kNeighbors:
        similarityRow = item_sims[itemID]
        for innerID, score in enumerate(similarityRow):
            candidates[innerID] += score * (rating / 5.0)

    watched = {itemID: 1 for itemID, _ in trainSet.ur[testUserInnerID]}
    recommendations = []

    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if itemID not in watched:
            movieID = trainSet.to_raw_iid(itemID)
            recommendations.append({
                "id": movieID,
                "name": ml.getMovieName(movieID),
                "score": round(ratingSum, 3)
            })
        if len(recommendations) >= 10:
            break

    return recommendations

def get_content_based_recommendations(testUserInnerID):
    """Get Content-Based Recommendations for a user with interest similarity boost."""
    k = 10  # Number of recommendations
    testUserRatings = trainSet.ur[testUserInnerID]  # Get user's past ratings

    if not testUserRatings:
        print(f"⚠️ No ratings found for user {testUserInnerID}, returning popular movies.")
        return get_popular_movies()


    user_interests = ml.loadUserInterests()
    genres = ml.getGenres()


    # Get raw UID (not internal ID) because interests use raw IDs
    try:
        raw_user_id = trainSet.to_raw_uid(testUserInnerID)
    except ValueError:
        raw_user_id = testUserInnerID
    raw_user_id = int(raw_user_id)

    user_interest_list = user_interests.get(raw_user_id, [])
    print(f"User Interests = {user_interest_list}")

    candidates = defaultdict(float)
    
    for itemID, rating in testUserRatings:
        similarityRow = contentKNN.similarities[itemID]

        for innerID, score in enumerate(similarityRow):
            if score > 0:
                boost = 0
                movieID = trainSet.to_raw_iid(innerID)
                movie_genres = genres.get(movieID, [])

                # ✅ Check if user interests match the first genre
               # Count how many genres overlap with user interests
                matching_genres = [g for g in movie_genres if g in user_interest_list]
                if matching_genres:
                    boost = 1 + 0.4 * (len(matching_genres) / len(movie_genres))  # Normalize boost by movie genres count


                    print("Boost applied", boost)
                    print("Movie genre", movie_genres)
                    print(f"Checking movie {movieID} genres: {movie_genres}, user interests: {user_interest_list}")

                    
                final_score = (score * (rating / 5.0)) * boost

                candidates[innerID] += final_score

    watched = {itemID for itemID, _ in testUserRatings}
    recommendations = []

    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if itemID not in watched:
            movieID = trainSet.to_raw_iid(itemID)
            name = ml.getMovieName(movieID)

            recommendations.append({
                "id": movieID,
                "name": name,
                "score": round(ratingSum, 3)
            })

        if len(recommendations) >= 10:
            break

    print(recommendations)
    return recommendations


def get_locations_by_theme(themes, top_n=10):
    if not hasattr(ml, 'genreIDs'):
        ml.getGenres()

    if isinstance(themes, str):
        themes = [themes]

    themes = [t.strip().lower() for t in themes]
    theme_freq = Counter(themes)

    genre_id_weights = {}
    for theme, freq in theme_freq.items():
        for genre, genre_id in ml.genreIDs.items():
            if genre.lower() == theme:
                genre_id_weights[genre_id] = freq
                break

    if not genre_id_weights:
        return []

    genres = ml.getGenres()
    movie_scores = {}

    for movieID, movie_genre_ids in genres.items():
        matched_weights = [genre_id_weights[gid] for gid in movie_genre_ids if gid in genre_id_weights]

        if matched_weights:
            # Sum of matched weights normalized by total genres count
            score = sum(matched_weights) / len(movie_genre_ids)
            movie_scores[movieID] = score

    top_movie_ids = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = [{
        "id": movieID,
        "name": ml.getMovieName(movieID),
        "score": score
    } for movieID, score in top_movie_ids]

    return recommendations

def get_popular_movies():
    """Simple fallback to popular movies."""
    popular_movie_ids = list(ml.movieID_to_name.keys())[:10]
    return [{
        "id": movieID,
        "name": ml.getMovieName(movieID),
        "score": 1.0  # Assume a default high score
    } for movieID in popular_movie_ids]


def reload_models():
    global data, trainSet, userKNN, itemKNN, user_sims, itemKNN, item_sims, contentKNN

    data = ml.loadMovieLensLatestSmall()
    trainSet = data.build_full_trainset()

    print("Updated trainSet:", trainSet)  # Print the updated trainSet for debugging

    # User-based model
    sim_options_user = {'name': 'cosine', 'user_based': True}
    userKNN = KNNBasic(sim_options=sim_options_user)
    userKNN.fit(trainSet)
    user_sims = userKNN.compute_similarities()

    # Item-based model
    sim_options_item = {'name': 'pearson', 'user_based': False}
    itemKNN = KNNBasic(sim_options=sim_options_item)
    itemKNN.fit(trainSet)
    item_sims = itemKNN.compute_similarities()

    contentKNN.fit(trainSet)



@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'})

@app.route('/prepare_user', methods=['POST'])
def prepare_user():
    print("📩 /prepare_user endpoint was called")
    data = request.json
    firebase_uid = data.get('firebase_uid')

    if not firebase_uid:
        return jsonify({'error': 'Missing firebase_uid'}), 400

    # Strictly ordered steps:
    generate_ratings_csv(firebase_uid)
    generate_user_info_xlsx(firebase_uid)
    reload_models()

    return jsonify({'status': 'prepared', 'uid': firebase_uid})



@app.route('/update_interests', methods=['POST'])
def update_interests():
    data = request.get_json()
    print("Received userId in Flask:", data.get("userId"))  # Debug log for userId
    user_id = data.get('userId')
    interests = data.get('interests')  # Get interests from the request
    if not user_id:
        return jsonify({'error': 'Missing userId'}), 400
    
    if interests is not None:
        # ✅ Step 1: Update user info first
        print("Updated Interests:", interests)
        generate_user_info_xlsx(user_id)

        # ✅ Step 2: THEN reload models based on updated files
        reload_models()

        return jsonify({'message': 'User interests updated successfully!'}), 200
    else:
        return jsonify({'error': 'Interests not provided'}), 400
    
    


@app.route('/add_review', methods=['POST'])
def add_review():
    data = request.get_json()
    user_id = data.get('userId')
    if not user_id:
        return jsonify({'error': 'Missing userId'}), 400

    generate_ratings_csv(user_id)
    reload_models()
    return jsonify({'message': 'ratings.csv updated'}), 200

@app.route('/generate', methods=['POST'])
def generate_files():
    print("📩 /generate endpoint was called")  # DEBUG
    data = request.json
    print("Data received:", data)  # Add this
    firebase_uid = data.get('firebase_uid')
    if not firebase_uid:
        return jsonify({'error': 'Missing firebase_uid'}), 400

    generate_ratings_csv(firebase_uid)
    generate_user_info_xlsx(firebase_uid)

    reload_models()  # <- Add this line here!

    return jsonify({'status': 'success', 'uid': firebase_uid})


@app.route('/get_numeric_user_id', methods=['POST'])
def get_numeric_id():
    data = request.json
    firebase_uid = data.get('firebase_uid')

    print(f"Received UID: {firebase_uid}") 
    
    if not firebase_uid:
        return jsonify({'error': 'Missing firebase_uid'}), 400
    
    numeric_id = get_numeric_user_id(firebase_uid)

    print(f"Mapped Numeric ID: {numeric_id}") 
    return jsonify({'user_id': int(numeric_id)})


print("Sample Content Similarity Matrix:")
print(contentKNN.similarities[:5, :5])  # Print a small portion

def ensemble_recommendations(user_recs, item_recs, content_recs, weights=(0.2, 0.3, 0.5)):
    combined_scores = defaultdict(float)
    all_recs = [user_recs, item_recs, content_recs]
    
    for rec_list, weight in zip(all_recs, weights):
        for rec in rec_list:
            combined_scores[rec['id']] += weight * rec['score']

    # Sort and get top 10
    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_recommendations = []
    for movie_id, score in sorted_recs[:10]:
        final_recommendations.append({
            "id": movie_id,
            "name": ml.getMovieName(movie_id),
            "score": round(score, 3)
        })
    
    return final_recommendations

@app.route('/similar_locations', methods=['GET'])
def get_similar_locations():
    theme = request.args.get('theme')
    
    # Example of querying similar locations based on theme
    similar_locations = get_locations_by_theme(theme)

    return jsonify({"locations": similar_locations})


@app.route('/recommendations', methods=['GET'])
def recommend():
    raw_user_id = request.args.get('user_id')
    print(f"Received user_id: {raw_user_id}")

    if not raw_user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    try:
        testUserInnerID = trainSet.to_inner_uid(str(raw_user_id))
        print(f"[{datetime.now().isoformat()}] ✅ Mapped to inner user ID: {testUserInnerID}")

        user_based_recs = get_user_based_recommendations(testUserInnerID)
        item_based_recs = get_item_based_recommendations(testUserInnerID)
        content_based_recs = get_content_based_recommendations(testUserInnerID)

    except ValueError:
        print(f"❄️ Cold-start: User ID {raw_user_id} not in training set.")

        # ✅ Support multiple themes from query (passed as JSON array or comma-separated string)
        theme_param = request.args.get('themes')
        themes = []

        if theme_param:
            try:
                themes = json.loads(theme_param) if theme_param.startswith('[') else [t.strip() for t in theme_param.split(',')]
            except Exception as e:
                print("⚠️ Error parsing themes:", e)
                themes = [theme_param.strip()]

        if themes:
            print(f"❄️ Cold-start fallback using themes: {themes}")
            fallback_recs = get_locations_by_theme(themes, top_n=10)
            return jsonify({
                "ensemble": fallback_recs,
                "content_based": [],
                "item_based": [],
                "user_based": [],
                "cold_start": True
            })

        return jsonify({'message': 'No recommendations available for this user.'}), 200

    # ✅ Normal (non-cold-start) case
    ensemble_recs = ensemble_recommendations(user_based_recs, item_based_recs, content_based_recs)

    if not ensemble_recs:
        return jsonify({'message': 'No recommendations available for this user.'}), 200

    return jsonify({
        "user_based": user_based_recs,
        "item_based": item_based_recs,
        "content_based": content_based_recs,
        "ensemble": ensemble_recs
    })




def print_content_based_recommendations(user_id):
    try:
        testUserInnerID = trainSet.to_inner_uid(user_id)
        recommendations = get_content_based_recommendations(testUserInnerID)
        
        if not recommendations:
            print(f"No recommendations found for user {user_id}.")
            return
        
        print(f"Top Content-Based Recommendations for User {user_id}:")
        for idx, rec in enumerate(recommendations, start=1):
            print(f"{idx}. {rec['name']} (Score: {rec['score']})")
    except ValueError:
        print(f"Error: User ID {user_id} not found in dataset.")

# Example usage
#print_content_based_recommendations('1013')

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=5000)
