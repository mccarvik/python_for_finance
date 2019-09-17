"""
module to perform k nearest neighbor
"""
import pdb
import math
from random import random, randint
import datetime as dt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.ml_utils import plot_decision_regions, IMG_PATH, IMG_ROOT, standardize
from research.ml_analysis.dev_work.optimization import anneal_opt, genetic_opt
mpl.use('Agg')

# NOTES: Throughout script assume last item in the dataset is the label

def wineprice(rating, age):
    """
    Simulate how a wine price will move
    """
    peak_age = rating - 50
    # Calculate price based on rating
    price = rating / 2
    if age > peak_age:
        # Past its peak, goes bad in 10 years
        price = price * (5 - (age - peak_age) / 2)
    else:
        # Increases to 5x original value as it
        # approaches its peak
        price = price * (5 * ((age + 1) / peak_age))
    if price < 0:
        price = 0
    return price


def wineset1():
    """
    Creates a bunch of reasonable prices with a little randomness thrown into the price
    """
    rows = []
    for _ in range(300):
        # Create a random age and rating
        rating = random()*50+50
        age = random()*50
        # Get reference price
        price = wineprice(rating, age)
        # Add some noise
        price *= (random() * 0.2 + 0.9)
        # Add to the dataset
        rows.append((rating, age, price))
    return np.array(rows)


def wineset2():
    """
    same as set 1 but adding irrelevant and heterogenous data
    bottlesize = heterogenous, aisle = irrelevant
    """
    rows = []
    for _ in range(300):
        rating = random() * 50 + 50
        age = random() * 50
        aisle = float(randint(1, 20))
        bottlesize = [375.0, 750.0, 1500.0][randint(0, 2)]
        price = wineprice(rating, age)
        price *= (bottlesize / 750)
        price *= (random() * 0.2 + 0.9)
        rows.append((rating, age, aisle, bottlesize, price))
    return np.array(rows)


def wineset3():
    """
    Uneven distribution - ex: "some bought wine at a discount"
    """
    rows = wineset1()
    for row in rows:
        if random() < 0.5:
            # Wine was bought at a discount store
            row[-1] *= 0.6
    return rows


class KNN():
    """
    K-Nearest neighbor Classifier
    """
    def __init__(self, data, k_nbrs=3, weight_func=False, knn_func=False, cat=False,
                 opt=False, opt_func=anneal_opt, dont_div=False):
        """
        Constructor function
        """
        self.k_nbrs = k_nbrs
        self.cat = cat
        self.data = data

        if weight_func:
            self.weightf = weight_func
        else:
            self.weightf = gaussian_wgt
        if knn_func:
            self.knn_func = knn_func
        else:
            self.knn_func = self.wgt_knn_est

        if opt:
            pdb.set_trace()
            wgt_domain = [(0, 20)] * (len(self.data[0])-1)
            costf = self.create_cost_func()
            scales = opt_func(wgt_domain, costf, step=2)
            self.data = self.rescale_data(scales)
        if not dont_div:
            self.train, self.test = self.div_data()
        else:
            self.train = self.data
            self.test = np.array([])

    def rescale_data(self, scale):
        """
        Rescale the data based on scale
        Helps eliminate irrelevant data and minimize outscaled heterogenous data
        """
        scaleddata = []
        for row in self.data:
            scaled = [scale[i] * row[i] for i in range(len(scale))]
            scaleddata.append(scaled + [row[-1]])
        return np.array(scaleddata)

    def div_data(self, test=0.3, data=np.array([])):
        """
        Divide data into test and train
        Not necessarily an even split, done on random number generator
        """
        trainset = []
        testset = []
        if data.size != 0:
            row_iter = data
        else:
            row_iter = self.data

        for row in row_iter:
            if random() < test:
                testset.append(row)
            else:
                trainset.append(row)
        return np.array(trainset), np.array(testset)

    def wgt_knn_est(self, vec1, target=False):
        """
        knn with the distances weighted for the outcome
        target parameter indicates if the data has the target attached or not
        """
        # Get distances
        # pdb.set_trace()
        if target:
            dlist = self.get_dist(vec1[:-1])
        else:
            dlist = self.get_dist(vec1)
        if not self.cat:
            avg = 0.0
        else:
            avg = {}

        totalweight = 0.0
        # Get weighted average
        for cnt in range(self.k_nbrs):
            dist = dlist[cnt][0]
            idx = dlist[cnt][1]
            weight = self.weightf(dist)
            # Adds up all the results and multiplies more for closer data points,
            # than averages over all the k nodes
            # Assumes label is the last item in the list
            if not self.cat:
                avg += weight * self.train[idx][-1]
            else:
                if self.train[idx][-1] in avg:
                    avg[self.train[idx][-1]] += weight
                else:
                    avg[self.train[idx][-1]] = weight
            totalweight += weight

        if totalweight == 0:
            return 0

        if not self.cat:
            # Need this as the weight might be more than 1
            # avg is average price not average dist
            avg = avg / totalweight
        else:
            # categorical --> weighted voting of neighbors
            avg = sorted(avg, key=(lambda key: avg[key]), reverse=True)[0]
        return avg

    def knn_est(self, vec1):
        """
        Gets the k closest neighbors and average of those results
        """
        # Get sorted distances
        dlist = self.get_dist(vec1)
        avg = 0.0

        # Take the average of the top k results
        for ind in range(self.k_nbrs):
            idx = dlist[ind][1]
            avg += self.train.ix[idx]['label']
        avg = avg / self.k_nbrs
        return avg

    def get_dist(self, vec1):
        """
        Calcs the distance from a the given vector compared with every other vector
        Returns a sorted list with the closest vectors at the top
        """
        distancelist = []
        # Loop over every item in the dataset
        idx = 0
        for row in self.train:
            vec2 = row[:-1]
            # Add the distance and the index
            distancelist.append((euclidean(vec1, vec2), idx))
            idx += 1

        # Sort by distance
        distancelist.sort()
        return distancelist

    def predict(self, vectors, target=False):
        """
        Predicts the outcome of a list of entries
        """
        ests = []
        for vec in vectors:
            ests.append(self.knn_func(vec, target))
        return np.array(ests)

    def run(self, trials=10, data=np.array([])):
        """
        run the algorithm
        """
        total_error = 0
        for cnt in range(trials):
            error = 0.0
            print("Trial number: {}".format(cnt+1))
            if trials != 1:
                if data.size != 0:
                    self.train, self.test = self.div_data(data=data)
                else:
                    self.train, self.test = self.div_data()
            guesses = self.predict(self.test, target=True)
            ind = 0
            for est in guesses:
                if not self.cat:
                    error += (self.test[ind][-1] - est)**2
                else:
                    if self.test[ind][-1] != est:
                        error += 1
                ind += 1
            print("test error: {}".format(error / len(self.test)))
            total_error += error / len(self.test)
        return total_error / trials

    def create_cost_func(self):
        """
        Creating a cost function to use for optimization
        """
        def costf(scale):
            """
            Shell for a function to be used as a cost function for optimization
            """
            sdata = self.rescale_data(scale)
            return self.run(trials=10, data=sdata)
        return costf

    def probguess(self, vec1, low, high):
        """
        Returns the probability that price is in the range given based on nearby nodes
        """
        dlist = self.get_dist(vec1)
        nweight = 0.0
        tweight = 0.0

        for ind in range(self.k_nbrs):
            # pdb.set_trace()
            dist = dlist[ind][0]
            idx = dlist[ind][1]
            weight = self.weightf(dist)
            val = self.data[idx][-1]

            # Is this point in the range?
            if val >= low and val <= high:
                nweight += weight
            tweight += weight
        if tweight == 0:
            return 0
        # The probability is the weights in the range divided by all the weights
        return nweight / tweight

    def cum_graph(self, vec1, high):
        """
        cumulative probability graph
        """
        time1 = np.arange(0.0, high, 0.1)
        cprob = np.array([self.probguess(vec1, 0, v) for v in time1])
        plt.plot(time1, cprob)
        plt.ylim(0, 1)
        plt.savefig(IMG_PATH + "cum_density_{}.png"
                    "".format(dt.datetime.now().strftime("%Y%m%d")))
        plt.close()

    def prob_graph(self, vec1, high, s_split=5.0):
        """
        probability distribution graph
        """
        # Make a range for the prices
        time1 = np.arange(0.0, high, 0.1)

        # Get the probabilities for the entire range
        probs = [self.probguess(vec1, v, v+0.1) for v in time1]

        # Smooth them by adding the gaussian of the nearby probabilites
        smoothed = []
        # for ind in range(len(probs)):
        for ind, _ in enumerate(probs):
            s_vals = 0.0
            # for ind2 in range(0, len(probs)):
            for ind2, _ in enumerate(probs):
                # farther away points have greater dist which mutes their weight
                dist = abs(ind - ind2) * 0.1
                weight = gaussian_wgt(dist, sigma=s_split)
                s_vals += weight * probs[ind2]
            smoothed.append(s_vals)
        smoothed = np.array(smoothed)

        plt.plot(time1, smoothed)
        plt.savefig(IMG_PATH + "prob_density_{}.png"
                    "".format(dt.datetime.now().strftime("%Y%m%d")))
        plt.close()

    def plot_knn(self, col1='col1', col2='col2', name="knn_custom_"):
        """
        plot the decision boundaries of this knn instance
        """
        plot_decision_regions(self.train[:, :-1], self.train[:, -1],
                              classifier=self)
        pdb.set_trace()
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.savefig(IMG_PATH + name +
                    "{}.png".format(dt.datetime.now().strftime("%Y%m%d")))
        plt.close()


def gaussian_wgt(dist, sigma=5.0):
    """
    weight is 1 when distance is 0
    weight never falls all the way to 0
    """
    return math.e**(-dist**2 / (2 * sigma**2))


def inverse_wgt(dist, num=1.0, const=0.1):
    """
    weight function - as dist gets smaller, so does denominator and weight grows
    """
    return num / (dist + const)


def subtract_wgt(dist, const=1.0):
    """
    subtracts distance from a constant, so smaller dist, higher weight
    if dist is greater than const, won't return any weight
    """
    if dist > const:
        return 0
    return const - dist


def euclidean(vec1, vec2):
    """
    Pythagorean Theorem for distances
    """
    dist = 0.0
    for ind, _ in enumerate(vec1):
        dist += (vec1[ind] - vec2[ind])**2
    return math.sqrt(dist)


def pml_knn_test():
    """
    Test our knn vs sklearn
    """
    # Get Data
    iris = datasets.load_iris()
    x_vals = iris.data[:, [2, 3]]
    y_vals = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals,
                                                        test_size=0.3, random_state=0)
    x_train_std = standardize(x_train)
    x_test_std = standardize(x_test)
    # x_combined = np.vstack((x_train, x_test))
    x_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))
    iris_data = np.concatenate((x_train_std, np.array([y_train]).T), axis=1)

    # Sklearn KNN
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(x_train_std, y_train)
    # x_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(x_combined_std, y_combined,
                          classifier=knn, test_break_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'knn_sklearn.png', dpi=300)
    plt.close()

    # Custom KNN
    cust_knn = KNN(iris_data, k_nbrs=5, dont_div=True)
    plot_decision_regions(x_combined_std, y_combined,
                          classifier=cust_knn, test_break_idx=range(105, 150))
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'knn_cust.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    pml_knn_test()
    # regression based data
    DATA = wineset3()
    DATA = wineset1()
    DATA = wineset2()
    # KNN_INST = KNN(np.array(DATA), opt=True, opt_func=genetic_opt)
    KNN_INST = KNN(np.array(DATA))
    # print(KNN_INST.run(10))

    # probability guess
    # print(KNN_INST.probguess([99, 20], 40, 80))
    # print(KNN_INST.probguess([99, 20], 80, 120))
    # print(KNN_INST.probguess([99, 20], 120, 1000))
    # print(KNN_INST.probguess([99, 20], 30, 120))
    # KNN_INST.cum_graph([1, 1], 120)
    # KNN_INST.prob_graph([99, 20], 120)

    # categorical data
    # IRIS = datasets.load_iris()
    # IRIS = pd.merge(pd.DataFrame(IRIS.data[:, :2]), pd.DataFrame(IRIS.target),
    #                 left_index=True, right_index=True)

    # KNN_INST = KNN(IRIS.values, cat=True)
    # print(KNN_INST.run(10))

    # plot_decision_regions(KNN_INST.train[:, :-1], KNN_INST.train[:, -1],
    #                       classifier=KNN_INST)
    # plt.xlabel('sepal length')
    # plt.ylabel('sepal width')
    # plt.savefig(IMG_PATH + "knn_custom_{}.png"
    #                       "".format(dt.datetime.now().strftime("%Y%m%d")))
    # plt.close()
