# from surprise import KNNBasic, NormalPredictor
# from ContentKNNAlgorithm import ContentKNNAlgorithm
# from Evaluator import Evaluator
# from MovieLens import MovieLens
# from EnsembleAlgorithm import EnsembleAlgorithm

# import random
# import numpy as np

# # Set seeds for reproducibility
# np.random.seed(0)
# random.seed(0)

# # Load dataset and rankings
# print("🎬 Loading MovieLens data...")
# ml = MovieLens()
# data = ml.loadMovieLensLatestSmall()
# rankings = ml.getPopularityRanks()

# # Create evaluator
# evaluator = Evaluator(data, rankings)

# # Add User-Based CF
# userKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
# evaluator.AddAlgorithm(userKNN, "UserKNN")

# # Add Item-Based CF
# itemKNN = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
# evaluator.AddAlgorithm(itemKNN, "ItemKNN")

# # Add Content-Based
# contentKNN = ContentKNNAlgorithm()
# evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# # Ensemble
# ensembleAlgo = EnsembleAlgorithm(userKNN, itemKNN, contentKNN, weights=(0.3, 0.2, 0.5))
# evaluator.AddAlgorithm(ensembleAlgo, "EnsembleKNN")

# # Add Random Predictor
# randomPredictor = NormalPredictor()
# evaluator.AddAlgorithm(randomPredictor, "RandomPredictor")

# # 🔍 Evaluate models
# evaluator.Evaluate(doTopN=False)

# # ✨ Show recommendations for a specific user
# user_id = str(1002)  # 👈 You can change the user ID here
# n_recommendations = 10  # Number of recommendations to show

# print(f"\n📜 Top {n_recommendations} recommendations for User {user_id} using EnsembleKNN:")

# # Train the ensemble algorithm on the full dataset
# ensembleAlgo.fit(data.build_full_trainset())

# # Get all items the user has not rated yet
# trainset = data.build_full_trainset()
# anti_testset = []

# user_inner_id = trainset.to_inner_uid(user_id)

# for item_inner_id in trainset.all_items():
#     if not trainset.knows_item(item_inner_id):
#         continue
#     if not trainset.ur[user_inner_id] or item_inner_id not in [iid for (iid, _) in trainset.ur[user_inner_id]]:
#         item_raw_id = trainset.to_raw_iid(item_inner_id)
#         anti_testset.append((user_id, item_raw_id, 0.0))

# # Predict scores
# predictions = ensembleAlgo.test(anti_testset)

# # Sort predictions
# predictions.sort(key=lambda x: x.est, reverse=True)

# # Display top-N
# for i in range(n_recommendations):
#     movie_id = predictions[i].iid
#     movie_name = ml.getMovieName(movie_id)
#     print(f"{i+1}. {movie_name} (Predicted Rating: {predictions[i].est:.2f})")

from surprise import KNNBasic, NormalPredictor
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from MovieLens import MovieLens
from EnsembleAlgorithm import EnsembleAlgorithm

import random
import numpy as np

# Set seeds for reproducibility
np.random.seed(0)
random.seed(0)

# Load dataset and rankings
print("🎬 Loading MovieLens data...")
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()
rankings = ml.getPopularityRanks()

# Create evaluator
evaluator = Evaluator(data, rankings)

# Add User-Based CF
userKNN = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(userKNN, "UserKNN")

# Add Item-Based CF
itemKNN = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
evaluator.AddAlgorithm(itemKNN, "ItemKNN")

# Add Content-Based
contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Ensemble
ensembleAlgo = EnsembleAlgorithm(userKNN, itemKNN, contentKNN, weights=(0.3, 0.2, 0.5))
evaluator.AddAlgorithm(ensembleAlgo, "EnsembleKNN")

# Add Random Predictor
randomPredictor = NormalPredictor()
evaluator.AddAlgorithm(randomPredictor, "RandomPredictor")

# 🔍 Evaluate models (prints RMSE, MAE for all algorithms)
evaluator.Evaluate(doTopN=False)

# ✨ Show recommendations for a specific user using ItemKNN
user_id = str(1002)  # 👈 You can change this user ID
n_recommendations = 10  # Number of recommendations to display

print(f"\n📜 Top {n_recommendations} recommendations for User {user_id} using ItemKNN:")

# Build the trainset
trainset = data.build_full_trainset()

# Train the ItemKNN algorithm on the full dataset
itemKNN.fit(trainset)

# Generate anti-testset (unrated items by the user)
anti_testset = []
user_inner_id = trainset.to_inner_uid(user_id)

for item_inner_id in trainset.all_items():
    if item_inner_id not in [iid for (iid, _) in trainset.ur[user_inner_id]]:
        item_raw_id = trainset.to_raw_iid(item_inner_id)
        anti_testset.append((user_id, item_raw_id, 0.0))

# Predict ratings for all items the user hasn't rated
item_predictions = itemKNN.test(anti_testset)

# Sort predictions by estimated rating in descending order
item_predictions.sort(key=lambda x: x.est, reverse=True)

# Display top-N recommendations
for i in range(n_recommendations):
    movie_id = item_predictions[i].iid
    movie_name = ml.getMovieName(movie_id)
    print(f"{i+1}. {movie_name} (Predicted Rating: {item_predictions[i].est:.2f})")
