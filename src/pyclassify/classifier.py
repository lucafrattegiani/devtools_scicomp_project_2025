from pyclassify.utils import distance, majority_vote
from typing import List, Tuple

class kNN:
    def __init__(self, k: int):
        """
        Initializes the kNN classifier with a specified number of nearest neighbors.
        
        Args:
            k (int): The number of nearest neighbors to consider.
        """

        # Check if k is an integer at runtime
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        
        # Check if k is a positive integer
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.k = k

    def _get_k_nearest_neighbors(self, X, y, x):
        """
        Finds the k nearest neighbors of the point x based on Euclidean distance.
        
        Args:
            X (list of lists): The dataset of points (features).
            y (list): The corresponding labels of the points in X.
            x (list): The new point to classify.
        
        Returns:
            list: A list of class labels of the k nearest neighbors of x.
        """
        distances = []
        for i in range(len(X)):
            dist = distance(x, X[i])  # Use the distance function from utils.py
            distances.append((dist, y[i]))  # Store the distance and corresponding label
        
        distances.sort(key=lambda x: x[0])  # Sort by distance
        k_nearest_neighbors = [label for _, label in distances[:self.k]]
        
        return k_nearest_neighbors

    def __call__(self, data: Tuple[List[List[float]], List[int]], new_points: List[List[float]]):
        """
        Classifies a list of new points by finding their k nearest neighbors and performing majority voting.
        
        Args:
            data (tuple): A tuple containing the feature matrix (X) and the labels (y).
            new_points (list): A list of points to classify.
        
        Returns:
            list: A list of predicted class labels for each point in new_points.
        """
        X, y = data  # Unpack the feature matrix and labels
        
        predictions = []  # To store the predicted classes for each new point
        
        for point in new_points:
            # Get the k nearest neighbors for the current point
            neighbors = self._get_k_nearest_neighbors(X, y, point)
            
            # Perform majority voting to get the predicted class
            predicted_class = majority_vote(neighbors)
            
            # Store the predicted class for the current point
            predictions.append(predicted_class)
        
        return predictions
