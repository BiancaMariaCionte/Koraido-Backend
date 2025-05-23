# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:25:39 2018

@author: Frank
"""


from ai.MovieLens import MovieLens
from ai.ContentKNNAlgorithm import ContentKNNAlgorithm
from ai.Evaluator import Evaluator
from surprise import NormalPredictor

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

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
