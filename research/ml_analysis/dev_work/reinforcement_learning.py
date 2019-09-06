"""
Module to implement Reinforcement Learning
"""
import pdb
import re
import math
import warnings
import feed_filter

DATA_PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/dev_work/dev_data/"

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_words(doc):
    """
    Breaks up a doc into words
    """
    splitter = re.compile('\\W*')
    # print(doc)
    # Split the words by non-alpha characters
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]

    # Return the unique set of words only
    return dict([(w, 1) for w in words])


class RLClassifier:
    """
    Will classify documents based on training data
    """
    def __init__(self, getfeatures, filename=None):
        # Counts of feature/category combinations
        self.f_c = {}
        # Counts of documents in each category
        self.cnt = {}
        self.getfeatures = getfeatures

    def incf(self, feat, cat):
        """
        Increases the count of a feature / category pair
        """
        self.f_c.setdefault(feat, {})
        self.f_c[feat].setdefault(cat, 0)
        self.f_c[feat][cat] += 1

    def incc(self, cat):
        """
        Increase the count of a category
        """
        self.cnt.setdefault(cat, 0)
        self.cnt[cat] += 1

    def fcount(self, feat, cat):
        """
        The number of times a feature has appeared in a category
        """
        if feat in self.f_c and cat in self.f_c[feat]:
            return float(self.f_c[feat][cat])
        return 0.0

    def catcount(self, cat):
        """
        The number of items in a category
        """
        if cat in self.cnt:
            return float(self.cnt[cat])
        return 0

    def totalcount(self):
        """
        calculate the total count of items
        """
        return sum(self.cnt.values())

    def categories(self):
        """
        The list of all categories
        """
        return self.cnt.keys()

    def train(self, item, cat):
        """
        Train the classifier
        """
        features = self.getfeatures(item)
        # Increment the count for every feature with this category
        for feat in features:
            self.incf(feat, cat)
        # Increment the count for this category
        self.incc(cat)

    def fprob(self, feat, cat):
        """
        P(f|cat) --> probability of f, given cat
        """
        if self.catcount(cat) == 0:
            return 0
        # The total number of times this feature appeared in this
        # category divided by the total number of items in this category
        return self.fcount(feat, cat) / self.catcount(cat)

    def weighted_prob(self, feat, cat, prf, weight=1.0, ass_prob=0.5):
        """
        Calculate the weighted probability of a category resulting in a given feature
        Uses an assumed probability as a starting point
        """
        # Calculate current probability
        basic_prob = prf(feat, cat)

        # Count the number of times this feature has appeared in all categories
        totals = sum([self.fcount(feat, c) for c in self.categories()])

        # Calculate the weighted average
        w_prob = ((weight * ass_prob) + (totals * basic_prob)) / (weight + totals)
        return w_prob


class NaiveBayesClassifier(RLClassifier):
    """
    Class to mimic a classifer based on Bayes Theorem
    """
    def __init__(self, getfeatures):
        """
        Constructor
        """
        RLClassifier.__init__(self, getfeatures)
        self.thresholds = {}

    def doc_prob(self, item, cat):
        """
        Calculates the unconditional probability of these features resulting in this cat
        Assumes the features are independent, ex:
            prob of X U Y = prob of X * prob of Y
        """
        features = self.getfeatures(item)

        # Multiply the probabilities of all the features together
        prob = 1
        for feat in features:
            # calc the weighted prob that this feature results in this cat
            # Multiply all probabilities together, assume independent events
            prob *= self.weighted_prob(feat, cat, self.fprob)
        return prob

    def prob(self, item, cat):
        """
        Calculates the probability of these items resulting in this category
        """
        # Bayes Theorem
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.doc_prob(item, cat)
        return docprob * catprob

    def setthreshold(self, cat, thresh):
        """
        Sets the threshold value
        """
        self.thresholds[cat] = thresh

    def getthreshold(self, cat):
        """
        return the threshold
        """
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def classify(self, item, default=None):
        """
        Classifies an item into a category
        """
        probs = {}
        # Find the category with the highest probability
        max_val = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max_val:
                max_val = probs[cat]
                best = cat

        # Make sure the probability exceeds threshold*next best
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                return default
        return best


class FisherClassifier(RLClassifier):
    """
    Classifier designed to calculate the porbability that the results are more
    or less likely than a random set
    """
    def __init__(self, getfeatures):
        """
        Constructor
        """
        RLClassifier.__init__(self, getfeatures)
        self.minimums = {}

    def cprob(self, feat, cat):
        """
        Calculates the odds this feature is in this category divided the odds it
        is in all categories
        Assumes there will be an equal number of items in each category
        """
        # The frequency of this feature in this category
        clf = self.fprob(feat, cat)
        if clf == 0:
            return 0

        # The frequency of this feature in all the categories
        freqsum = sum([self.fprob(feat, c) for c in self.categories()])
        # The probability is the frequency in this category divided by the overall frequency
        prob = clf / (freqsum)
        # odds its in this category / sum of odds its in each category
        return prob

    def fisherprob(self, item, cat):
        """
        If probabilites are independent and random, result would fit Chi^2 distribution
        By Feading result to the invesrse Chi^2 we get the probability that a
        random set would return such a high number
        """
        # Multiply all the probabilities together
        prob = 1
        features = self.getfeatures(item)
        for feat in features:
            prob *= (self.weighted_prob(feat, cat, self.cprob))

        # Take the natural log and multiply by -2
        fscore = -2 * math.log(prob)
        # Use the inverse chi2 function to get a probability
        return invchi2(fscore, len(features) * 2)

    def setminimum(self, cat, min_val):
        """
        Sets the minimum value for this category
        """
        self.minimums[cat] = min_val

    def getminimum(self, cat):
        """
        Returns minimum value for this category
        """
        if cat not in self.minimums:
            return 0
        return self.minimums[cat]

    def classify(self, item, default=None):
        """
        Calculates the probabilites for each category and determines the best
        result that exceeds the specified minimum
        """
        # Loop through looking for the best result
        best = default
        max_val = 0.0
        for cat in self.categories():
            prob = self.fisherprob(item, cat)
            # Make sure it exceeds its minimum
            if prob > self.getminimum(cat) and prob > max_val:
                best = cat
                max_val = prob
        return best


def sampletrain(classifier):
    """
    Function to dump some sample training data for convenience
    """
    classifier.train('Nobody owns the water.', 'good')
    classifier.train('the quick rabbit jumps fences', 'good')
    classifier.train('buy pharmaceuticals now', 'bad')
    classifier.train('make quick money at the online casino', 'bad')
    classifier.train('the quick brown fox jumps', 'good')


def invchi2(chi, deg_free):
    """
    Inverse Chi Squared function
    """
    m_val = chi / 2.0
    tot_sum = term = math.exp(-m_val)
    for i in range(1, deg_free//2):
        term *= m_val / i
        tot_sum += term
    return min(tot_sum, 1.0)


if __name__ == '__main__':
    RLC = RLClassifier(get_words)

    # First demo
    # RLC.train('the quick brown fox jumps over the lazy dog', 'good')
    # RLC.train('make quick money in the online casino', 'bad')
    # print(RLC.fcount('quick', 'good'))
    # print(RLC.fcount('quick', 'bad'))

    # Weighted Prob ex
    # sampletrain(RLC)
    # print(RLC.weighted_prob('money', 'good', RLC.fprob))

    # Naive Bayes Classifier
    # NBC = NaiveBayesClassifier(get_words)
    # sampletrain(NBC)
    # print(NBC.prob('quick rabbit', 'good'))
    # print(NBC.prob('quick rabbit', 'bad'))
    # print(NBC.classify('quick rabbit', default='unknown'))
    # print(NBC.classify('quick money', default='unknown'))
    # NBC.setthreshold('bad', 3.0)
    # print(NBC.classify('quick money', default='unknown'))

    # Fisher Classifier
    FISHC = FisherClassifier(get_words)
    sampletrain(FISHC)
    # print(FISHC.cprob('money', 'bad'))
    # print(FISHC.weighted_prob('money', 'bad', FISHC.cprob))
    # print(FISHC.cprob('quick', 'good'))
    # print(FISHC.fisherprob('quick rabbit', 'good'))
    # print(FISHC.fisherprob('quick rabbit', 'bad'))
    # print(FISHC.classify('quick rabbit'))
    # print(FISHC.classify('quick money'))
    # FISHC.setminimum('bad', 0.8)
    # print(FISHC.classify('quick money'))
    # FISHC.setminimum('good', 0.5)
    # print(FISHC.classify('quick money'))
    
    # Using Feed Filter
    feed_filter.read('python_search.xml', FISHC)
