from surprise import AlgoBase

class EnsembleAlgorithm(AlgoBase):
    def __init__(self, user_algo, item_algo, content_algo, weights=(0.2, 0.2, 0.6)):
        AlgoBase.__init__(self)
        self.user_algo = user_algo
        self.item_algo = item_algo
        self.content_algo = content_algo
        self.weights = weights
        self.trainset = None  # Initialize trainset as None

    def fit(self, trainset):
        """Fit all three models on the same training set."""
        self.trainset = trainset  # Store the trainset
        self.user_algo.fit(trainset)
        self.item_algo.fit(trainset)
        self.content_algo.fit(trainset)
        return self

    def estimate(self, u, i):
        """Estimate a rating for a user-item pair."""
        est = 0.0
        total_weight = 0.0

        for algo, weight in zip([self.user_algo, self.item_algo, self.content_algo], self.weights):
            try:
                est += algo.estimate(u, i) * weight
                total_weight += weight
            except:
                continue  # Skip if this algorithm fails to estimate

        # Return the weighted average, or fallback to 3.0 if no valid predictions
        return est / total_weight if total_weight > 0 else 3.0
