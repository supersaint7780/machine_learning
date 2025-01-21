import numpy as np

def confusion_matrix(y_true, y_predicted):
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_predicted[i]] += 1
    return confusion_matrix

def plot_confusion_matrix(y_true, y_predicted, labels):
    confusion_matrix = confusion_matrix(y_true, y_predicted)
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(confusion_matrix, annot=True, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
