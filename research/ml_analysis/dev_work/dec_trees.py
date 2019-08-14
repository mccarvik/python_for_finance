"""
Module for creating Decision Trees
"""
# import pdb
from math import log

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


class DecisionTree():
    """
    Class to mimic a Decision Tree
    """
    def __init__(self, col=-1, value=None, results=None, true_n=None, false_n=None):
        """
        col = column to be tested
        value = val that must match to get true result
        tb and fb = next nodes in the tree, true and false
        results = dictionary of results for this branch
        """
        self.col = col
        self.value = value
        self.results = results
        self.true_n = true_n
        self.false_n = false_n


def divideset(rows, column, value):
    """
    Divides a set on a specific column. Can handle numeric or nominal values
    Make a function that tells us if a row is in
    the first group (true) or the second group (false) and return the sets
    """
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        # If the value is numerical, use "greater than" to split data
        split_function = lambda row: row[column] >= value
    else:
        # If the value is not numerical, use boolean "equals" to split data
        split_function = lambda row: row[column] == value
    # Divide the rows into two sets and return them
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


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


def gini_impurity(rows):
    """
    error rate if one of the results in the set is randomly applied to one of
    the items in the set
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


# Entropy is the sum of p(x)log(p(x)) across all the different possible results
def entropy(rows):
    """
    The amount of disorder in a set
    Calculates the frequency of each item: p(i) = count(i) / count(total)
    then calculates entropy: sum of all ( p(i) * log(p(i)) )
    """
    log2 = lambda x: log(x) / log(2)
    results = unique_cts(rows)
    # Now calculate the entropy
    ent = 0.0
    for res in results:
        prob = float(results[res]) / len(rows)
        ent = ent - prob * log2(prob)
    return ent
