from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import os
import argparse
import random

def split_dataset(X, y, train_percent=0.8, shuffle = True):
    """
    Split the dataset into training and testing sets based on the given percentage.
    Args:
        X (list): Feature matrix.
        y (list): Labels.
        train_percent (float): Percentage of the data to be used for training (default 80%).
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """

    #Data shuffling:    
    if shuffle:
        # Combine X and y into a single list of tuples (X, y) for easier shuffling
        data = list(zip(X, y))

        random.shuffle(data)
    
        # Unzip the shuffled dataset back in
        X[:], y[:] = zip(*data)

    # Calculate the number of training samples
    split_index = int(len(X) * train_percent)
    
    # Split the dataset
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    
    return X_train, y_train, X_test, y_test


def compute_accuracy(predictions, actual_labels):
    """
    Compute the accuracy of the kNN model.
    Args:
        predictions (list): List of predicted labels.
        actual_labels (list): List of actual labels.
    Returns:
        float: Accuracy as a percentage.
    """
    correct = sum(1 for pred, actual in zip(predictions, actual_labels) if pred == actual)
    accuracy = (correct / len(actual_labels)) * 100
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    config_path = args.config

    # Step 1: Read the parameters from the config.yaml file
    config = read_config(config_path)  # path to config.yaml
    k = config.get('k', 5)  # Number of neighbors (default is 5)
    dataset_path = config.get('dataset', './data/ionosphere.data')  # Path to dataset
    backhand = config.get("backhand")

    # Step 2: Load the dataset
    X, y = read_file(dataset_path)

    # Step 3: Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
    X_train, y_train, X_test, y_test = split_dataset(X, y)

    # Step 4: Initialize the kNN classifier with k from the config
    model = kNN(k=k, backhand=backhand)

    # Step 5: Perform classification on the test set
    predictions = model(data=(X_train, y_train), new_points=X_test)

    # Step 6: Compute accuracy
    accuracy = compute_accuracy(predictions, y_test)
    print(f"Accuracy: {accuracy:.2f}%")
    print("K: " + str(k))

    # Step 7: Add, commit, and push changes to Git with message "practical2"
    # os.system("git add .")
    # os.system('git commit -m "practical2"')
    # os.system("git push")

if __name__ == "__main__":
    main()

