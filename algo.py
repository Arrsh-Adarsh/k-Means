
# class for K-Means cluster

import numpy as np
import pandas as pd
class K_Means:

    # constructor for initializing variable
    def __init__(self, k=8, tolerance=0.0001, max_iter=300):
        self.classes = {}
        self.centroids = {}
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter

    # fit method for training of dataset
    def fit(self, X):

        # initializing centroids of the cluster
        j = 0
        for i in range(self.k):

            while X[j] in X[:j]:
                j += 1
            self.centroids[i] = X[j]
            j += 1

        # Entering the main loop
        for i in range(self.max_iter):
            for i in range(self.k):
                self.classes[i] = []

            # calculating centroid for each data
            for features in X:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                cluster = distances.index(min(distances))
                self.classes[cluster].append(features)

            previous_centroid = self.centroids

            # updating values of centroids in cluster
            for cluster in self.classes: self.centroids[cluster] = np.average(self.classes[cluster], axis=0)

            isOptimal = True

            # checking for tolerance value
            for centroid in self.centroids:
                original_centroid = previous_centroid[centroid]
                curr_centroid = self.centroids[centroid]

                if np.sum((curr_centroid - original_centroid) / original_centroid * 100.0) < self.tolerance:
                    isOptimal = False

            if isOptimal: break

    # method for prediction of new values
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        cluster = distances.index(min(distances))
        return cluster