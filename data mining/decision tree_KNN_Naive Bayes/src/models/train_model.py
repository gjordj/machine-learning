from numpy import loadtxt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import graphviz

import pickle
from matplotlib import pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


def train_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(criterion="entropy",
                                         random_state=100)
    # max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy = clf_entropy.fit(X_train, y_train)

    file_path_columns_names = '../../data/processed/columns_names.pickle'
    open_file = open(file_path_columns_names, "rb")
    features = list(pickle.load(open_file))
    open_file.close()
    print("Features Names", features)

    return clf_entropy, features


def load_train_validation_sets():
    file_path_X_test = '../../data/processed/X_test.csv'  # comes from the Training set.
    file_path_X_train = '../../data/processed/X_train.csv'
    file_path_y_train = '../../data/processed/y_train.csv'
    file_path_y_test = '../../data/processed/y_test.csv'

    X_test = loadtxt(file_path_X_test, delimiter=',')
    X_train = loadtxt(file_path_X_train, delimiter=',')

    y_train = loadtxt(file_path_y_train, delimiter=',')
    y_test = loadtxt(file_path_y_test, delimiter=',')
    return X_train, X_test, y_train, y_test


def load_vehicle_train_validation_sets():
    file_path_X = '../../data/interim/vehicle/X.csv'
    file_path_Y = '../../data/interim/vehicle/Y.csv'
    file_path_X_test = '../../data/interim/vehicle/X_test.csv'
    file_path_X_train = '../../data/interim/vehicle/X_train.csv'
    file_path_y_train = '../../data/interim/vehicle/y_train.csv'
    file_path_y_test = '../../data/interim/vehicle/y_test.csv'

    X = loadtxt(file_path_X, delimiter=',')
    Y = loadtxt(file_path_Y, delimiter=',')

    X_test = loadtxt(file_path_X_test, delimiter=',')
    X_train = loadtxt(file_path_X_train, delimiter=',')

    y_train = loadtxt(file_path_y_train, delimiter=',')
    y_test = loadtxt(file_path_y_test, delimiter=',')
    return X, Y, X_train, X_test, y_train, y_test


def performance_metrics(y_test, y_pred, model_name):
    print(f"\nThis is the performance of the {model_name}:")
    print("Confusion Matrix:\n ",
          confusion_matrix(y_test, y_pred))
    print("Accuracy:\n ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report:\n ",
          classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred) * 100


def check_sets(X_train, X_validation, y_train, y_validation):
    print('X_train:\n', X_train)
    print('X_validation:\n', X_validation)
    print('y_train:\n', y_train)
    print('y_validation:\n', y_validation)


def traverse_tree_structure(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )


def plot_and_analyze_tree(decision_tree_model_entropy, features, tree_type):
    traverse_tree_structure(decision_tree_model_entropy)
    filename = f'../../reports/{tree_type}.dot'
    # To see the tree:
    #    https://bit.ly/3qk2VGK
    tree_data = tree.export_graphviz(decision_tree_model_entropy,
                                     out_file=filename,
                                     feature_names=features,
                                     class_names=['Low', 'High'],
                                     filled=True)
    # Draw graph
    graph = graphviz.Source(tree_data, format="png")
    graph
    text_representation = tree.export_text(decision_tree_model_entropy)
    print(text_representation)


def store_model(model, name_file):
    filename = f'../../models/{name_file}.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_model(name_file):
    filename = f'../../models/{name_file}.sav'
    model = pickle.load(open(filename, 'rb'))
    return model


def decision_tree_without_pruning_sklearn():
    X_train, X_validation, y_train, y_validation = load_train_validation_sets()

    # check_sets(X_train, X_validation, y_train, y_validation)

    file_path_columns_names = '../../data/processed/columns_names.pickle'
    open_file = open(file_path_columns_names, "rb")
    features = list(pickle.load(open_file))
    open_file.close()
    print("Features Names", features)

    decision_tree_model_entropy, features = train_using_entropy(X_train, y_train)

    store_model(decision_tree_model_entropy, 'decision_tree_model_entropy')
    decision_tree_model_entropy = load_model('decision_tree_model_entropy')

    y_pred = decision_tree_model_entropy.predict(X_validation)
    performance_metrics(y_validation, y_pred, 'decision tree without pruning')
    plot_and_analyze_tree(decision_tree_model_entropy, features, 'tree_not_pruned')


def load_validation_set():
    file_path_X_validation = '../../data/processed/X_validation.csv'
    file_path_y_validation = '../../data/processed/y_validation.csv'
    X_validation = loadtxt(file_path_X_validation, delimiter=',')
    y_validation = loadtxt(file_path_y_validation, delimiter=',')
    return X_validation, y_validation


def train_naive_bayes_smoothing_and_store(X_train, y_train):
    # alpha: 1. Additive (Laplace/Lidstone) smoothing parameter.
    naive_bayes_model_smoothing = ComplementNB(alpha=1)
    naive_bayes_model_smoothing.fit(X_train, y_train)
    store_model(naive_bayes_model_smoothing, 'naive_bayes_model_smoothing')


def naive_bayes_train_and_performance(naive_bayes_model_smoothing, X_validation, y_validation):
    y_pred = naive_bayes_model_smoothing.predict(X_validation)
    performance_metrics(y_validation, y_pred, 'Naive Bayes with Smoothing')


def replace_item(x):
    if x == 1:
        return 'High'
    else:
        return 'Low'


def naive_bayes_test_prediction(naive_bayes_model_smoothing):
    file_path_test_set_naive_bayes = '../../data/processed/x_test_set_naive_bayes.csv'
    X_test_set_naive_bayes = loadtxt(file_path_test_set_naive_bayes, delimiter=',')
    y_pred = naive_bayes_model_smoothing.predict(X_test_set_naive_bayes)

    print("The prediction made by the Naive Bayes model with Smoothing is: ", y_pred)

    file_path_test_set_naive_bayes = '../../data/raw/test_set_naive_bayes.csv'
    X_test_set_naive_bayes = pd.read_csv(file_path_test_set_naive_bayes)

    y_pred = [replace_item(x) for x in y_pred]
    X_test_set_naive_bayes['Salary Prediction'] = y_pred
    print(X_test_set_naive_bayes)
    return y_pred


def naive_bayes_classifier():
    X_train, X_validation, y_train, y_validation = load_train_validation_sets()  # it uses the test.csv... from
    # training set
    train_naive_bayes_smoothing_and_store(X_train, y_train)
    naive_bayes_model_smoothing = load_model('naive_bayes_model_smoothing')
    naive_bayes_train_and_performance(naive_bayes_model_smoothing, X_validation, y_validation)
    naive_bayes_test_prediction(naive_bayes_model_smoothing)


def fit_and_store_knn_model_sklearn(K, normalized_x_train, y_train):
    """

    Args:
        K: k-neighbors
        normalized_x_train: normalized predictors to train the model.
        y_train: encoded target variable/label to be predicted during the model training.

    Returns:
        .sav file: model trained with the previous Args.

    """
    # training the model, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors
    # .KNeighborsClassifier.html

    # 'uniform' means unweighted
    print("K used!", K)
    knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean', weights='uniform')
    knn.fit(normalized_x_train, y_train)
    print("These are the parameters of the model:", knn.get_params())
    # save the model to disk
    filename = f'../../models/knn_trained_{K}_sklearn.sav'
    pickle.dump(knn, open(filename, 'wb'))


def normalize_df(x_train, x_validation):
    """
    Note: Note how the conversion of x_train using a function which fits (figures out the params) then normalises.
    Whilst the x_validation conversion just transforms, using the same params that it learned from the train data.

    Args:
        x_train: training predictors
        x_validation: test predictors

    Returns:
        normalized_x_train: normalized x_train
        normalized_x_validation: normalized x_validation


    """
    scaler = Normalizer().fit(x_train)  # the scaler is fitted to the training set
    normalized_x_train = scaler.transform(x_train)  # the scaler is applied to the training set
    normalized_x_validation = scaler.transform(x_validation)  # the scaler is applied to the validation set

    return normalized_x_train, normalized_x_validation


def normalize_X(X):
    """
    Note: Note how the conversion of x_train using a function which fits (figures out the params) then normalises.
    Whilst the x_validation conversion just transforms, using the same params that it learned from the train data.

    Args:
        x_train: training predictors
        x_validation: test predictors

    Returns:
        normalized_x_train: normalized x_train
        normalized_x_validation: normalized x_validation


    """
    scaler = Normalizer().fit(X)  # the scaler is fitted to the training set
    normalized_X = scaler.transform(X)  # the scaler is applied to the training set

    return normalized_X


def predict_calculate_accuracy(normalized_x_validation, y_validation, k):
    # load the model from disk
    filename = f'../../models/knn_trained_{k}_sklearn.sav'
    loaded_knn_model = pickle.load(open(filename, 'rb'))
    # predict based on the predictors
    y_pred_knn_classifier = loaded_knn_model.predict(normalized_x_validation)
    performance_metrics(y_validation, y_pred_knn_classifier, 'KNN model, without LOOCV')


def fit_knn_model_sklearn_loocv(X, Y, k):
    normalized_X = normalize_X(X)
    # create loocv procedure
    cv = LeaveOneOut()
    # enumerate splits
    y_validation, y_pred = list(), list()
    for train_ix, validation_ix in cv.split(normalized_X):
        # split data
        X_train, X_test = normalized_X[train_ix, :], normalized_X[validation_ix, :]
        y_train, y_test = Y[train_ix], Y[validation_ix]

        # fit model
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        knn_model.fit(X_train, y_train)

        # evaluate model
        yhat = knn_model.predict(X_test)

        # store
        y_validation.append(y_test[0])
        y_pred.append(yhat[0])
    return y_validation, y_pred


def load_features_correlation():
    file_path_dict_correlation = '../../data/interim/vehicle/features_correlation.pickle'
    open_file = open(file_path_dict_correlation, "rb")
    features_correlation = pickle.load(open_file)
    open_file.close()
    return features_correlation.reset_index(drop=True)


def load_feature_correlation_and_vehicle_df():
    filename = f'../../data/interim/vehicle/vehicle_df.pickle'
    vehicle_df = pickle.load(open(filename, 'rb')).reset_index(drop=True)
    features_correlation = load_features_correlation()
    print("Features ordered, from highest to lowest absolute correlation: \n", features_correlation)

    features_ordered_by_correlation = list(features_correlation[features_correlation.columns[0]])
    total_features = len(features_ordered_by_correlation)
    return total_features, features_ordered_by_correlation, vehicle_df


def accuracy_of_knn_model_sklearn_loocv_by_num_top_correlated_features(k, Y):
    accuracy_scores = {}
    total_features, features_ordered_by_correlation, vehicle_df = load_feature_correlation_and_vehicle_df()
    for num_feature in range(total_features):
        features_selected = features_ordered_by_correlation[0:total_features - num_feature]  # from 36 features to 0.
        X = vehicle_df[features_selected].values
        y_validation, y_pred_knn_classifier_loocv = fit_knn_model_sklearn_loocv(X, Y, k)
        # performance_metrics(y_validation, y_pred_knn_classifier_loocv, 'KNN model, with LOOCV')
        accuracy = accuracy_score(y_validation, y_pred_knn_classifier_loocv) * 100
        accuracy_scores[str(len(features_selected)) + ' features'] = round(accuracy, 2)
    num_features_accuracy = {k: v for k, v in sorted(accuracy_scores.items(), key=lambda item: item[1],
                                                     reverse=True)}
    return num_features_accuracy


def knn_classifier_filter_feature_selection(LOOCV, k, normalized_x_train, y_validation, y_train, Y):
    if LOOCV != True:
        fit_and_store_knn_model_sklearn(k, normalized_x_train, y_train)
        predict_calculate_accuracy(normalized_x_validation, y_validation, k)
    else:
        num_features_vs_accuracy = accuracy_of_knn_model_sklearn_loocv_by_num_top_correlated_features(k, Y)
        print(num_features_vs_accuracy)


def seq_feature_selection_features_and_accuracy(k, total_features, normalized_X, Y, vehicle_df):
    dict_num_features_accuracies = {}
    # from 1 feature to 35 features. Not 36 because we wouldn't perform any selection.
    for n_features_to_select in range(1, total_features):
        # Sequential Feature Selection
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
        sfs = SequentialFeatureSelector(knn_model, n_features_to_select=n_features_to_select)
        sfs.fit(normalized_X, Y)
        normalized_X_seq_features_selected = sfs.transform(normalized_X)

        y_validation, y_pred_knn_classifier_loocv = fit_knn_model_sklearn_loocv(normalized_X_seq_features_selected,
                                                                                Y,
                                                                                k)
        accuracy = accuracy_score(y_validation, y_pred_knn_classifier_loocv) * 100
        print("Number of features selected: ", n_features_to_select)
        print("Selected features: ", sfs.get_feature_names_out(vehicle_df.columns[:-1]))
        print("LOOCV's Accuracy: ", round(accuracy, 2), "%\n")

        dict_num_features_accuracies[str(n_features_to_select) + ' features'] = round(accuracy, 2)
    dict_num_features_accuracies = {k: v for k, v in
                                    sorted(dict_num_features_accuracies.items(), key=lambda item: item[1],
                                           reverse=True)}
    print(dict_num_features_accuracies)
    return dict_num_features_accuracies


def knn_classifier_seq_feature_selection(k, Y):
    total_features, features_ordered_by_correlation, vehicle_df = load_feature_correlation_and_vehicle_df()
    X = vehicle_df[vehicle_df.columns[:-1]].values
    normalized_X = normalize_X(X)
    seq_feature_selection_features_and_accuracy(k, total_features, normalized_X, Y, vehicle_df)


def knn_classifier(LOOCV):
    X, Y, X_train, X_validation, y_train, y_validation = load_vehicle_train_validation_sets()
    normalized_x_train, normalized_x_validation = normalize_df(X_train, X_validation)
    k = 7
    knn_classifier_filter_feature_selection(LOOCV, k, normalized_x_train, y_validation, y_train, Y)
    knn_classifier_seq_feature_selection(k, Y)


# Decision Tree Functions
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


# Column labels.
# These are used only to print the tree.
header = ["Level", "Career",
          "Years of Experience", "Salary"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


#######
# Demo:
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


#######
# Demo:
# class_counts(training_data)
#######


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


#######
# Demo:
# is_numeric(7)
# is_numeric("Red")
#######


class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.
    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch, gain):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.gain = gain


def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch, gain)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))
    print(spacing + ' Attribute gain:' + str(format(node.gain, '.2f')))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def load_data():
    file_path = f'../../data/raw/training_set.csv'
    df = pd.read_csv(file_path)
    del df['Instance']
    df = df.to_numpy()
    print(df)
    return df


def decision_tree_without_pruning():
    training_data = load_data()
    my_tree = build_tree(training_data)
    print_tree(my_tree)


# def decision_tree_with_pruning():
#     # to train the model
#     # it should contain all the rows of the first table. Not the 0.7 of rows.
#     file_path_X = '../../data/processed/X.csv'
#     X_train = loadtxt(file_path_X, delimiter=',')
#     file_path_y = '../../data/processed/y.csv'
#     y_train = loadtxt(file_path_y, delimiter=',')
#
#     # X_train, X_validation, y_train, y_validation = load_train_validation_sets()  # test files
#     # check_sets(X_train, X_validation, y_train, y_validation)
#
#     file_path_columns_names = '../../data/processed/columns_names.pickle'
#     open_file = open(file_path_columns_names, "rb")
#     features = list(pickle.load(open_file))
#     open_file.close()
#     print("Features Names", features)
#
#     # decision_tree_model_entropy, features = train_using_entropy(X_train, y_train)
#
#     clf = DecisionTreeClassifier()
#     path = clf.cost_complexity_pruning_path(X_train, y_train)  # should contain all the rows, not 0.7
#     path
#     ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(ccp_alphas, impurities)
#     plt.xlabel("effective alpha")
#     plt.ylabel("total impurity of leaves")
#     plt.show()
#     clfs = []
#
#     for ccp_alpha in ccp_alphas:
#         clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
#         clf.fit(X_train, y_train)
#         clfs.append(clf)
#
#     tree_depths = [clf.tree_.max_depth for clf in clfs]
#     plt.figure(figsize=(10, 6))
#     plt.plot(ccp_alphas[:-1], tree_depths[:-1])
#     plt.xlabel("effective alpha")
#     plt.ylabel("total depth")
#     plt.show()
#
#     from sklearn.metrics import accuracy_score
#
#     # y_validation, X_Validation .csv files
#     file_path_X_test = '../../data/processed/X_validation.csv'
#     X_test = loadtxt(file_path_X_test, delimiter=',')
#     file_path_y_test = '../../data/processed/y_validation.csv'
#     y_test = loadtxt(file_path_y_test, delimiter=',')
#     acc_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]
#
#     tree_depths = [clf.tree_.max_depth for clf in clfs]
#     plt.figure(figsize=(10, 6))
#     plt.grid()
#     plt.plot(ccp_alphas[:-1], acc_scores[:-1])
#     plt.xlabel("effective alpha")
#     plt.ylabel("Accuracy scores")
#     plt.show()

    # to validate the model: take last data from pdf
    # read X_validation file in the right format.
    # y_pred = decision_tree_model_entropy.predict(X_validation)

    # performance_metrics(y_validation, y_pred, 'decision tree without pruning')
    # plot_and_analyze_tree(decision_tree_model_entropy, features, 'tree_not_pruned')


def main():
    decision_tree_without_pruning()
    # To see the tree:
    #    https://bit.ly/3qk2VGK
    decision_tree_without_pruning_sklearn()
    # decision_tree_with_pruning()
    naive_bayes_classifier()
    LOOCV = True
    knn_classifier(LOOCV)


if __name__ == '__main__':
    main()
