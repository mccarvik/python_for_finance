"""
Module for creating Decision Trees
"""
import pdb
import datetime as dt
from math import log
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from utils.ml_utils import IMG_PATH, IMG_ROOT, plot_decision_regions

mpl.use('Agg')

DATA = [['slashdot', 'USA', 'yes', 18, 'None'],
        ['google', 'France', 'yes', 23, 'Premium'],
        ['digg', 'USA', 'yes', 24, 'Basic'],
        ['kiwitobes', 'France', 'yes', 23, 'Basic'],
        ['google', 'UK', 'no', 21, 'Premium'],
        ['(direct)', 'New Zealand', 'no', 12, 'None'],
        ['(direct)', 'UK', 'no', 21, 'Basic'],
        ['google', 'USA', 'no', 24, 'Premium'],
        ['slashdot', 'France', 'yes', 19, 'None'],
        ['digg', 'USA', 'no', 18, 'None'],
        ['google', 'UK', 'no', 18, 'None'],
        ['kiwitobes', 'UK', 'no', 19, 'None'],
        ['digg', 'New Zealand', 'yes', 12, 'Basic'],
        ['slashdot', 'UK', 'no', 21, 'None'],
        ['google', 'UK', 'yes', 18, 'Basic'],
        ['kiwitobes', 'France', 'yes', 19, 'Basic']]

HOUSING_DATA = []
with open("/home/ec2-user/environment/python_for_finance/research/ml_analysis/dev_work/dev_data/housing_data.csv") as file:
    LINES = file.readlines()
    FIRST = True
    CASTS = []
    for line in LINES:
        if FIRST:
            CASTS = line
            FIRST = False
            continue
        HOUSING_DATA.append(line.strip().split(','))
HOUSING_DATA = pd.DataFrame(HOUSING_DATA)
for typ in enumerate(CASTS.strip().split(",")):
    if typ[1] == 'Y':
        HOUSING_DATA[typ[0]] = pd.to_numeric(HOUSING_DATA[typ[0]])
HOUSING_DATA = HOUSING_DATA.values


###################
# Goal is to minimuze both gini impurity and entropy
# They both peak when result outcomes are relatively evenly distributed
###################


def gini_impurity(rows):
    """
    Measures the disorder of the set
    Error rate if one of the results in the set is randomly applied to one of
    the other rows in the set
    ie if the set is not very "disordered" this new result will equal the
    result that was previously there and vice versa
    """
    total = len(rows)
    counts = unique_cts(rows)
    imp = 0
    # loop through each result
    for ind1 in counts:
        # get percentage of total rows this result is
        prob1 = float(counts[ind1]) / total
        for ind2 in counts:
            if ind1 == ind2:
                continue
            # get percentage of total rows result 2 is
            prob2 = float(counts[ind2]) / total
            # add to total impurity by multiplying the two
            imp += prob1 * prob2
    return imp


def entropy(rows):
    """
    The amount of disorder in a set
    Calculates the frequency of each item: p(i) = count(i) / count(total)
    then calculates entropy: sum of all ( p(i) * log(p(i)) )
    entropy peaks slower than gini impurity but has overall similar shape
    """
    # This will always be negative as x will always be 0<x<1 and if x<e,
    # then ln(x) will be negative divided by the constant ln(2)
    log2 = lambda x: log(x) / log(2)
    results = unique_cts(rows)
    # Now calculate the entropy
    ent = 0.0
    for res in results:
        # probability of res
        prob = float(results[res]) / len(rows)
        # log2 will be negative and larger with smaller prob
        # prob * log2(prob) peaks aroung 0.37
        ent = ent - prob * log2(prob)
    return ent


def variance(rows):
    """
    calculates the variance of the data to be used as a proxy for entropy
    """
    if rows.size == 0:
        return 0
    data = [float(row[-1]) for row in rows]
    mean = sum(data) / len(data)
    var = sum([(d - mean)**2 for d in data]) / len(data)
    return var


def error(prob):
    """
    Calculates the error of a decision tree output
    """
    return 1 - np.max([prob, 1 - prob])


class DecisionTree():
    """
    Class to mimic a Decision Tree
    """
    def __init__(self, col=-1, value=None, results=None, t_branch=None, f_branch=None, score_func=entropy, data=np.array([])):
        """
        col = column to be tested
        value = val that must match to get true result
        tb and fb = next nodes in the tree, true and false
        results = dictionary of results for this branch
        """
        if data.size > 0:
            self.score_func = score_func
            tree = self.build_tree(data)
            self.col = tree.col
            self.value = tree.value
            self.results = tree.results
            self.t_branch = tree.t_branch
            self.f_branch = tree.f_branch
        else:
            self.col = col
            self.value = value
            self.results = results
            self.t_branch = t_branch
            self.f_branch = f_branch
            self.score_func = score_func

    def classify(self, obs, node=None):
        """
        Classify a new observation based on the tree
        """
        if not node:
            node = self
        if node.results:
            return node.results
        else:
            val = obs[node.col]
            branch = None
            if isinstance(val, (float, int)):
                if val >= node.value:
                    branch = node.t_branch
                else:
                    branch = node.f_branch
            else:
                if val == node.value:
                    branch = node.t_branch
                else:
                    branch = node.f_branch
        return self.classify(obs, branch)

    def predict(self, obs):
        """
        Classify a list of observations
        """
        preds = []
        for ind_ob in obs:
            preds.append([*self.classify(ind_ob)][0])
        return np.array(preds)

    def prune(self, mingain=0.1, node=None):
        """
        Remove branches that are not substantially improving performance
        Avoids overfitting
        """
        if not node:
            node = self
        # If the branches aren't leaves, then prune them
        if not node.t_branch.results:
            node.prune(mingain, node.t_branch)
        if not node.f_branch.results:
            node.prune(mingain, node.f_branch)

        # If both the subbranches are now leaves, see if they should merged
        if node.t_branch.results and node.f_branch.results:
            # Build a combined dataset
            true_b, false_b = [], []
            for val, col in node.t_branch.results.items():
                true_b += [[val]] * col
            for val, col in node.f_branch.results.items():
                false_b += [[val]] * col

            # Test the reduction in entropy
            # entropy together vs avg entropy of both individually
            delta = entropy(true_b + false_b) - (entropy(true_b) + entropy(false_b) / 2)

            # If information gain isnt over the threshold, make branches None
            # And make results the combined of the two branches
            if delta < mingain:
                node.t_branch, node.f_branch = None, None
                node.results = unique_cts(true_b + false_b)

    def build_tree(self, rows, node=None):
        """
        Function that will build a DecisionTree
        """
        if not node:
            node = self
        current_score = self.score_func(rows)

        # Set up some variables to track the best criteria
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        # Go through each column in the dataset
        for col in range(0, len(rows[0])-1):
            column_values = {}
            for row in rows:
                # set each possible outcome for the column = 1
                column_values[row[col]] = 1
            for value in column_values:
                # 1. divide the rows up for each value in this column
                # set 1 - where given col = given value,   set 2 - where it does not
                # For numerical values - loops through all values in dataset and
                # separates >= and < for each value and checks for best gain
                (set1, set2) = divide_set(rows, col, value)

                # 2. See how much Information gain from this split
                # Information Gain = weighted average of two sets
                prob = float(len(set1)) / len(rows)
                gain = (current_score - prob * self.score_func(set1)
                        - (1 - prob) * self.score_func(set2))
                if gain > best_gain and set1.size > 0 and set2.size > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        # Create the sub branches
        if best_gain > 0:
            # find the children nodes
            true_branch = node.build_tree(best_sets[0], node=node)
            false_branch = node.build_tree(best_sets[1], node=node)
            return DecisionTree(col=best_criteria[0], value=best_criteria[1],
                                t_branch=true_branch, f_branch=false_branch,
                                score_func=self.score_func)
        # base case when no more information to be gained
        return DecisionTree(results=unique_cts(rows))

    def print_tree(self, indent='    ', level=0, node=None):
        """
        Display the tree in a readable format
        """
        if not node:
            node = self
        # Is this a leaf node?
        line_indent = indent * level
        if node.results != None:
            print(line_indent + str(node.results))
        else:
            # Print the criteria
            level += 1
            print(line_indent + str(node.col) + ':' + str(node.value) + '? ')
            # Print the branches
            print(line_indent + 'T->')
            self.print_tree(indent, level, node.t_branch)
            print(line_indent + 'F->')
            self.print_tree(indent, level, node.f_branch)

    def missing_data_classify(self, obs, node=None):
        """
        Classify data if some information may be missing
        """
        if not node:
            node = self

        if node.results:
            return node.results
        else:
            val = obs[node.col]
            if not val:
                # Dont have the value for this node
                # Get data from both sub branches
                t_rets = self.missing_data_classify(obs, node.t_branch)
                f_rets = self.missing_data_classify(obs, node.f_branch)
                # Get count of instances on each branch and use to calc weight
                tcount = sum(t_rets.values())
                fcount = sum(f_rets.values())
                t_wgt = float(tcount) / (tcount + fcount)
                f_wgt = float(fcount) / (tcount + fcount)
                result = {}
                # Create a result for this outcome and pass up the recursion tree
                for key, ret in t_rets.items():
                    result[key] = ret * t_wgt
                for key, ret in f_rets.items():
                    result[key] = ret * f_wgt
                return result
            else:
                # if we have the data, traverse the tree as normal
                if isinstance(val, (int, float)):
                    if val >= node.value:
                        branch = node.t_branch
                    else:
                        branch = node.f_branch
                else:
                    if val == node.value:
                        branch = node.t_branch
                    else:
                        branch = node.f_branch
            return self.missing_data_classify(obs, branch)


def divide_set(rows, column, value):
    """
    Divides a set on a specific column. Can handle numeric or nominal values
    Make a function that tells us if a row is in
    the first group (true) or the second group (false) and return the sets
    """
    split_function = None
    if isinstance(value, (float, int)):
        # If the value is numerical, use "greater than" to split data
        split_function = lambda row: row[column] >= value
    else:
        # If the value is not numerical, use boolean "equals" to split data
        split_function = lambda row: row[column] == value
    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (np.array(set1), np.array(set2))


def unique_cts(rows):
    """
    Returns a dict of the possible results and how often they occur
    """
    results = {}
    for row in rows:
        # The result is the last column
        res = row[-1]
        if res not in results:
            results[res] = 0
        results[res] += 1
    return results


def get_width(tree):
    """
    Get the width of the tree
    """
    if not tree.t_branch and not tree.f_branch:
        return 1
    return get_width(tree.t_branch) + get_width(tree.f_branch)


def get_depth(tree):
    """
    Get the depth of the tree
    """
    if not tree.t_branch and not tree.f_branch:
        return 0
    return max(get_depth(tree.t_branch), get_depth(tree.f_branch)) + 1


def draw_tree(tree, jpeg='tree_{}.jpg'):
    """
    Draw the tree structure in a jpeg
    """
    width = get_width(tree) * 100
    height = get_depth(tree) * 100 + 120
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, width / 2, 20)
    img.save(IMG_PATH + jpeg.format(dt.datetime.now().strftime("%Y%m%d")), 'JPEG')


def draw_node(draw, tree, x_val, y_val):
    """
    Draw the individual node of the tree
    """
    if not tree.results:
        # Get the width of each branch
        width1 = get_width(tree.f_branch) * 100
        width2 = get_width(tree.t_branch) * 100

        # Determine the total space required by this node
        left = x_val - (width1 + width2) / 2
        right = x_val + (width1 + width2) / 2

        # Draw the condition string
        draw.text((x_val-20, y_val-10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x_val, y_val, left+width1/2, y_val+100), fill=(255, 0, 0))
        draw.line((x_val, y_val, right-width2/2, y_val+100), fill=(255, 0, 0))

        # Draw the branch nodes
        draw_node(draw, tree.f_branch, left+width1/2, y_val+100)
        draw_node(draw, tree.t_branch, right-width2/2, y_val+100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
        draw.text((x_val-20, y_val), txt, (0, 0, 0))


def pml_build_tree_test():
    """
    Test our decision tree vs sklearn
    """
    # Get Data
    iris = datasets.load_iris()
    x_vals = iris.data[:, [2, 3]]
    y_vals = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals,
                                                        test_size=0.3, random_state=0)
    iris_data = np.concatenate((x_train, np.array([y_train]).T), axis=1)

    # Sklearn Tree
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(x_train, y_train)
    x_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(x_combined, y_combined,
                          classifier=tree, test_break_idx=range(105, 150))
    export_graphviz(tree,
                    out_file='tree.dot',
                    feature_names=['petal length', 'petal width'])

    # Custom Tree
    iris_tree = DecisionTree(data=iris_data)
    iris_tree.print_tree()
    x_comb = np.vstack((x_train, x_test))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(x_comb, y_comb,
                          classifier=iris_tree, test_break_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'decision_tree_decision.png', dpi=300)
    plt.close()
    draw_tree(iris_tree)


if __name__ == '__main__':
    pml_build_tree_test()
    # Testing functionality
    # print(divide_set(DATA, 2, 'yes'))
    # print(gini_impurity(DATA))
    # print(entropy(DATA))

    # Tree with category classification
    # TREE = build_tree(DATA)
    # TREE = DecisionTree(data=DATA)
    # TREE.print_tree()
    # draw_tree(TREE)
    # TREE.prune(1)
    # print_tree(TREE)
    # print(TREE.classify(['(direct)', 'USA', 'yes', 5]))
    # print(TREE.missing_data_classify(['google', 'France', None, None]))

    # Tree with numerical classification
    # HOUSE_TREE = DecisionTree(data=HOUSING_DATA, score_func=variance)
    # pdb.set_trace()
    # HOUSE_TREE.print_tree()
    # draw_tree(HOUSE_TREE, jpeg='house_tree_{}.jpg')
