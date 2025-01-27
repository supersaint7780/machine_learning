import numpy as np

def entropy(y):
    """Calculate the entropy of a dataset"""
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(X_column, y, threshold):
    """
    Calculate information gain for a split on a feature
    X_column: The feature column to split on
    y: The target column
    threshold: The threshold value for splitting
    """
    left_indices = X_column <= threshold
    right_indices = X_column > threshold

    # Calculate weighted average entropy after the split
    n = len(y)
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    weighted_avg_entropy = (len(y[left_indices]) / n) * left_entropy + (len(y[right_indices]) / n) * right_entropy

    # Calculate information gain
    return entropy(y) - weighted_avg_entropy

def find_best_split(X, y):
    """
    Find the best feature and threshold to split on
    X: Features dataset
    y: Target column
    """
    best_feature = None
    best_threshold = None
    best_info_gain = -1

    for feature in range(X.shape[1]):  # Loop over each feature
        X_column = X[:, feature]
        thresholds = np.unique(X_column)

        for threshold in thresholds:
            gain = information_gain(X_column, y, threshold)
            if gain > best_info_gain:
                best_feature = feature
                best_threshold = threshold
                best_info_gain = gain

    return best_feature, best_threshold

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        """
        Fit the training data to build the decision tree
        X: Features dataset
        y: Target column
        depth: Current depth of the tree
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if n_classes == 1 or n_samples <= 1 or (self.max_depth is not None and depth >= self.max_depth):
            most_common_class = np.argmax(np.bincount(y))
            return most_common_class

        # Find the best feature and threshold to split on
        best_feature, best_threshold = find_best_split(X, y)

        if best_feature is None:  # If no split is found, return the majority class
            most_common_class = np.argmax(np.bincount(y))
            return most_common_class

        # Split the data into left and right subsets
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        # Recursively build the tree
        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)

        # Return a dictionary representation of the tree
        self.tree = {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }
        return self.tree

    def predict_one(self, x, tree):
        """Predict a single sample"""
        if not isinstance(tree, dict):
            return tree

        feature = tree["feature"]
        threshold = tree["threshold"]

        if x[feature] <= threshold:
            return self.predict_one(x, tree["left"])
        else:
            return self.predict_one(x, tree["right"])

    def predict(self, X):
        """Predict multiple samples"""
        return [self.predict_one(x, self.tree) for x in X]