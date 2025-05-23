# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from ai.MovieLens import MovieLens
from surprise import KNNBasic
from surprise import NormalPredictor
from ai.Evaluator import Evaluator
from ai.ContentKNNAlgorithm import ContentKNNAlgorithm
from ai.EnsembleAlgorithm import EnsembleAlgorithm

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options = {'name': 'pearson', 'user_based': False})
evaluator.AddAlgorithm(ItemKNN, "Item KNN")

# Add Content-Based
# contentKNN = ContentKNNAlgorithm()
# evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Ensemble
# ensembleAlgo = EnsembleAlgorithm(UserKNN, ItemKNN, contentKNN, weights=(0.2, 0.2, 0.6))
# evaluator.AddAlgorithm(ensembleAlgo, "EnsembleKNN")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
