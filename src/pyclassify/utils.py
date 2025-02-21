from typing import List
import os
import yaml
from line_profiler import profile
import numpy as np
from ._numba_module import _distance_numba
from numpy import ndarray


@profile
def distance(point1: List[float], point2: List[float]) -> float:
    """
    Computes the square of the Euclidean distance between two points.
    
    Args:
        point1 (List[float]): The first point.
        point2 (List[float]): The second point.
    
    Returns:
        float: The squared Euclidean distance between point1 and point2.
    """
    if len(point1) != len(point2):
        raise ValueError("Unmatching dimensions")
    
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

@profile
def distance_numba(point1: ndarray, point2: ndarray) -> float:
    """
    Computes the square of the Euclidean distance between two points.
    
    Args:
        point1 (List[float]): The first point.
        point2 (List[float]): The second point.
    
    Returns:
        float: The squared Euclidean distance between point1 and point2.
    """
    if len(point1) != len(point2):
        raise ValueError("Unmatching dimensions")
    
    if not isinstance(point1, np.ndarray) and not isinstance(point2, np.ndarray):
        raise TypeError("You should pass a numpy array")
    
    return _distance_numba(point1, point2)

@profile
def majority_vote(neighbors: List[int]):
    """
    Returns the majority class among all the labels that we find in the list passed as argument, which stores the labels of all the k-nearest neighbors.
    """
    #Dictionary of labels:
    labels = {}

    for label in neighbors:
        if label in labels:
            labels[label] += 1
        else:
            labels[label] = 1

    # Find the class label with the maximum count
    majority_class = None
    max_count = 0
    for label, count in labels.items():
        if count > max_count:
            majority_class = label
            max_count = count
    
    return majority_class

@profile
def distance_numpy(point1: ndarray, point2: ndarray) -> float:
    """
    Computes the square of the Euclidean distance between two points.
    
    Args:
        point1 (List[float]): The first point.
        point2 (List[float]): The second point.
    
    Returns:
        float: The squared Euclidean distance between point1 and point2.
    """
    if len(point1) != len(point2):
        raise ValueError("Unmatching dimensions")
    
    if not isinstance(point1, np.ndarray) and not isinstance(point2, np.ndarray):
        raise TypeError("You should pass a numpy array")
    
    dist = np.sum((point1 - point2)**2)
    return dist

def read_config(file):
    filepath = os.path.abspath(f'{file}.yaml')
    with open(filepath, 'r') as stream:
        kwargs = yaml.safe_load(stream)
    return kwargs

def convert(x):
    if x == "g" or x == "1":
        x = 1
    else:
        x = 0
    return x

def read_file(filepath):
    """
    Reads the Ionosphere dataset file and returns the features (X) and labels (y).
    
    Args:
        filepath (str): Path to the dataset file (e.g., 'data/ionosphere.data').
        
    Returns:
        tuple: A tuple containing two lists:
            - X (list of lists): A list of feature vectors (34 continuous attributes for each instance).
            - y (list): A list of class labels ('g' or 'b') for each instance.
    """
    X = []  # List to store features (X)
    y = []  # List to store labels (y)
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Strip any leading/trailing whitespace and split by commas
                parts = line.strip().split(',')
                
                # Features are the first 34 columns
                features = list(map(float, parts[:-1]))  # Convert attributes to float
                label = parts[-1]  # The label is the last column
                
                # Append to the respective lists
                X.append(features)
                y.append(convert(label))
                
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
    except Exception as e:
        print(f"Error reading the file: {e}")
    
    return X, y
