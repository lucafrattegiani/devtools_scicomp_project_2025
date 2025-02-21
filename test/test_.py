# tests/test.py
import pytest
from pyclassify.utils import distance
from pyclassify.utils import majority_vote

def test_distance():
    # Test case 1: Two points with identical coordinates, distance should be 0
    point1 = [1.0, 2.0]
    point2 = [1.0, 2.0]
    assert distance(point1, point2) == 0.0  # Distance should be 0
    
    # Test case 2: Simple 2D points (distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2))
    point1 = [1.0, 2.0]
    point2 = [4.0, 6.0]
    expected_distance = ( (4.0 - 1.0) ** 2 + (6.0 - 2.0) ** 2 )  # Without the square root, since your function returns the square of the distance
    assert distance(point1, point2) == expected_distance
    
    # Test case 3: Different dimensions (3D distance example)
    point1 = [1.0, 2.0, 3.0]
    point2 = [4.0, 5.0, 6.0]
    expected_distance = ( (4.0 - 1.0) ** 2 + (5.0 - 2.0) ** 2 + (6.0 - 3.0) ** 2 )
    assert distance(point1, point2) == expected_distance

def test_majority_vote():
    # Test case 1: Simple majority (0 is the majority)
    neighbors = [1, 0, 0, 0]
    assert majority_vote(neighbors) == 0  # Majority vote is 0
    
    # Test case 2: Tie between two classes (for an even number of neighbors, the first class in the list wins)
    neighbors = [1, 0, 1, 0]
    assert majority_vote(neighbors) == 1  # Majority vote is 1 (since it's even, the first majority wins)
    
    # Test case 3: Single class (all labels are the same)
    neighbors = [0, 0, 0, 0]
    assert majority_vote(neighbors) == 0  # Majority vote is 0
    
    # Test case 4: Another case with a tie, but with different numbers
    neighbors = [2, 3, 2, 3, 3]
    assert majority_vote(neighbors) == 3  # Majority vote is 3

# tests/test.py
from pyclassify.classifier import kNN

def test_knn_constructor():
    # Test case 1: Valid k (positive integer)
    model = kNN(3)  # Initialize with a valid value of k
    assert model.k == 3  # Check that k is correctly set

    # Test case 2: Invalid value for k (string, should raise TypeError)
    with pytest.raises(TypeError):
        model = kNN("3")  # k should be an integer, not a string

    # Test case 3: Invalid k (float)
    with pytest.raises(TypeError):
        kNN(3.5)  # k should be an integer, not a float
    
    # Test case 4: Invalid k (negative value)
    with pytest.raises(ValueError):
        kNN(-1)  # k cannot be negative (depending on your use case)
    
    # Test case 5: Edge case (k = 0)
    with pytest.raises(ValueError):
        kNN(0)  # k should be at least 1 for meaningful kNN operation

    #Asserts on 'backhand' parameter:
    model1 = kNN(k = 3, backhand = "plain")
    assert model1.backhand == "plain"

    model2 = kNN(k = 3, backhand = "numpy")
    assert model2.backhand == "numpy"

    #Test 'backhand': Value different from 'plain' and 'numpy'
    with pytest.raises(ValueError):
        kNN(k = 3, backhand = "prova")