# import os
# import csv
# import sys
# import re

# from surprise import Dataset
# from surprise import Reader
# from collections import defaultdict
# import numpy as np
# import csv
# import os
# import pandas as pd

# class MovieLens:
#         def __init__(self):
#                 base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to ai/
#                 data_dir = os.path.join(base_dir, "ml-latest-small")  # ml-latest-small inside ai folder
#                 self.ratingsPath = os.path.join(data_dir, "ratings.csv")
#                 self.moviesPath = os.path.join(data_dir, "movies.csv")
#                 self.userInfoPath = os.path.join(data_dir, "user_info.xlsx")
#                 self.movieID_to_name = {}
#                 self.name_to_movieID = {}
        
#         def loadMovieLensLatestSmall(self):
#                 self.movieID_to_name = {}
#                 self.name_to_movieID = {}
        
#                 reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
        
#                 ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        
#                 with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
#                     movieReader = csv.reader(csvfile)
#                     next(movieReader)  # Skip header line
#                     for row in movieReader:
#                         movieID = row[0]  # Keep it as a string
#                         movieName = row[1]
#                         self.movieID_to_name[movieID] = movieName
#                         self.name_to_movieID[movieName] = movieID
        
#                 return ratingsDataset

#         def getUserRatings(self, user):
#                 userRatings = []
#                 hitUser = False
#                 with open(self.ratingsPath, newline='', encoding='ISO-8859-1') as csvfile:
#                     ratingReader = csv.reader(csvfile)
#                     next(ratingReader)
#                     for row in ratingReader:
#                         userID = int(row[0])  # User ID stays as an integer
#                         if user == userID:
#                             movieID = row[1]  # Keep as string
#                             rating = float(row[2])
#                             userRatings.append((movieID, rating))
#                             hitUser = True
#                         if hitUser and user != userID:
#                             break
        
#                 return userRatings

# #count how many times each movie was rated
#         def getPopularityRanks(self):
#                 ratings = defaultdict(int)
#                 rankings = defaultdict(int)
#                 with open(self.ratingsPath, newline='', encoding='ISO-8859-1') as csvfile:
#                     ratingReader = csv.reader(csvfile)
#                     next(ratingReader)
#                     for row in ratingReader:
#                         movieID = row[1]  # Keep as string
#                         ratings[movieID] += 1
#                 rank = 1
#                 for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
#                     rankings[movieID] = rank
#                     rank += 1
#                 return rankings
    
  
#         def getMovieRatings(self):
#                 ratings = {}
#                 with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
#                     movieReader = csv.reader(csvfile)
#                     next(movieReader)
#                     for row in movieReader:
#                         movieID = row[0]
#                         rating = float(row[3])  # Assuming rating is in the 4th column
#                         ratings[movieID] = rating
#                 return ratings
   
  
    
#         def getGenres(self):
#                 genres = defaultdict(list)
#                 self.genreIDs = {}  # Store genre IDs globally
#                 maxGenreID = 0
        
#                 with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
#                     movieReader = csv.reader(csvfile)
#                     next(movieReader)  # Skip header line
#                     for row in movieReader:
#                         movieID = row[0]  # Keep as string
#                         genreList = row[2].split('|')  # Assuming genres are in the 3rd column
#                         genreIDList = []
#                         for genre in genreList:
#                             if genre in self.genreIDs:
#                                 genreID = self.genreIDs[genre]
#                             else:
#                                 genreID = maxGenreID
#                                 self.genreIDs[genre] = genreID
#                                 maxGenreID += 1
#                             genreIDList.append(genreID)
#                         genres[movieID] = genreIDList
#                 return genres
    
    
#         def loadUserInterests(self):
#                 """Ensure user interests match genre IDs by using the same dictionary."""
#                 user_interests = defaultdict(list)
        
#                 # Ensure genres are already processed
#                 if not hasattr(self, 'genreIDs'):
#                     self.getGenres()  # Generate genre ID mappings if not already loaded
        
#                 df = pd.read_excel(self.userInfoPath)
        
#                 for _, row in df.iterrows():
#                     userID = int(row['userId'])
#                     interestList = row['interests'].split('|') if pd.notna(row['interests']) else []
#                     interestIDList = []
                
#                     for interest in interestList:
#                         if interest in self.genreIDs:  # Ensure consistency
#                             interestIDList.append(self.genreIDs[interest])  # Use genre ID mapping
        
#                     user_interests[userID] = interestIDList  # Assign matched genre IDs to user interests
                
                
#                 return user_interests
    
#         def getMovieName(self, movieID):
#                 return self.movieID_to_name.get(movieID, "")
        
#         def getMovieID(self, movieName):
#                 return self.name_to_movieID.get(movieName, "")

from services.firebase_config import db
from surprise import Dataset, Reader
from collections import defaultdict
import pandas as pd

class MovieLens:

        def __init__(self):
                self.movieID_to_name = {}
                self.name_to_movieID = {}
                self.genreIDs = {}

        def loadMovieLensLatestSmall(self):
                # Load ratings from Firestore
                ratings_ref = db.collection('ratings').stream()
                rating_data = []
                for doc in ratings_ref:
                    entry = doc.to_dict()
                    rating_data.append([
                        str(entry['userId']), 
                        str(entry['movieId']), 
                        float(entry['rating']), 
                        int(entry['timestamp'])
                    ])
        
                reader = Reader(line_format='user item rating timestamp', sep=',')
                ratingsDataset = Dataset.load_from_df(
                    pd.DataFrame(rating_data, columns=["user", "item", "rating", "timestamp"]),
                    reader
                )
        
                # Load movie info from Firestore instead of CSV
                movies_ref = db.collection('movies').stream()
                for doc in movies_ref:
                    movie_id = doc.id
                    movie_data = doc.to_dict()
                    title = movie_data.get('title', '')
                    self.movieID_to_name[movie_id] = title
                    self.name_to_movieID[title] = movie_id
        
                return ratingsDataset

        def getUserRatings(self, user):
                userRatings = []
                ratings_ref = db.collection('ratings').where('userId', '==', int(user)).stream()
                for doc in ratings_ref:
                    entry = doc.to_dict()
                    userRatings.append((entry['movieId'], float(entry['rating'])))
                return userRatings

        def getPopularityRanks(self):
                ratings = defaultdict(int)
                rankings = defaultdict(int)
        
                ratings_ref = db.collection('ratings').stream()
                for doc in ratings_ref:
                    entry = doc.to_dict()
                    movieID = entry['movieId']
                    ratings[movieID] += 1
        
                rank = 1
                for movieID, count in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
                    rankings[movieID] = rank
                    rank += 1
                return rankings

        def getGenres(self):
                genres = defaultdict(list)
                maxGenreID = 0
        
                # Load movie genres from Firestore
                movies_ref = db.collection('movies').stream()
                for doc in movies_ref:
                    movieID = doc.id
                    movie_data = doc.to_dict()
                    genreList = movie_data.get('genres', '').split('|')
                    genreIDList = []
                    for genre in genreList:
                        if genre not in self.genreIDs:
                            self.genreIDs[genre] = maxGenreID
                            maxGenreID += 1
                        genreIDList.append(self.genreIDs[genre])
                    genres[movieID] = genreIDList
        
                return genres

        def loadUserInterests(self):
                user_interests = defaultdict(list)
                if not self.genreIDs:
                    self.getGenres()
        
                users_ref = db.collection('user_info').stream()
                for doc in users_ref:
                    data = doc.to_dict()
                    userID = int(doc.id)  # Using Firestore document ID as userId
                    interestList = data.get('interests', '').split('|')
                    interestIDList = [self.genreIDs[interest] for interest in interestList if interest in self.genreIDs]
                    user_interests[userID] = interestIDList
        
                return user_interests

        def getMovieName(self, movieID):
                return self.movieID_to_name.get(movieID, "")

        def getMovieID(self, movieName):
                return self.name_to_movieID.get(movieName, "")

        def getMovieRatings(self):
                ratings = {}
                movies_ref = db.collection('movies').stream()
                for doc in movies_ref:
                    movieID = doc.id
                    movie_data = doc.to_dict()
                    if 'rating' in movie_data:
                        ratings[movieID] = float(movie_data['rating'])
                return ratings

