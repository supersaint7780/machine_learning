import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from decision_tree import DecisionTree

IRIS_DATA_PATH = './dataset/iris.csv'
iris_df = pd.read_csv(IRIS_DATA_PATH)

# convert categorical values to numerical values
iris_df['species'] = iris_df['species'].map({
    'Iris-setosa': 0, 
    'Iris-versicolor': 1, 
    'Iris-virginica': 2
})

# split the data into features and target
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris_df['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

tree = DecisionTree(max_depth=3)
print(tree.fit(X_train, y_train))

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Predictions: {y_pred}")
print(f"Actual Labels: {y_test}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))