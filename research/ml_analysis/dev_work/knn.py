"""
module to perform k nearest neighbor
"""
import pdb
import math
import sys
from random import random, randint
import datetime as dt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
sys.path.append("/home/ec2-user/environment/python_for_finance/")
from utils.ml_utils import plot_decision_regions, IMG_ROOT
from optimization import anneal_opt, genetic_opt

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
    """
    rows = wineset1()
    for row in rows:
        if random() < 0.5:
            # Wine was bought at a discount store
            row['result'] *= 0.6
    return rows


class KNN():
    """
    K-Nearest neighbor Classifier
    """
    def __init__(self, data, k_nbrs=3, weight_func=False, knn_func=False, cat=False, opt=False, opt_func=anneal_opt):
        """
        Constructor function
        """
        self.k_nbrs = k_nbrs
        self.cat = cat
        self.data = data
        
        if weight_func:
            self.weightf = weight_func
        else:
            self.weightf = self.gaussian_wgt
        if knn_func:
            self.knn_func = knn_func
        else:
            self.knn_func = self.wgt_knn_est
        
        pdb.set_trace()
        if opt:
            wgt_domain = [(0, 10)] * 4
            costf = self.create_cost_func()
            scales = opt_func(wgt_domain, costf, step=2)
            pdb.set_trace()
            self.data = self.rescale_data(scales)
        self.train, self.test = self.div_data()

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
        divide data into test and train
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

    def wgt_knn_est(self, vec1):
        """
        knn with the distances weighted for the outcome
        """
        # Get distances
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

    def euclidean(self, vec1, vec2):
        """
        Pythagorean Theorem for distances
        """
        dist = 0.0
        for ind in range(len(vec1)):
            dist += (vec1[ind] - vec2[ind])**2
        return math.sqrt(dist)

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
            distancelist.append((self.euclidean(vec1[:-1], vec2), idx))
            idx += 1

        # Sort by distance
        distancelist.sort()
        return distancelist

    def inverse_wgt(self, dist, num=1.0, const=0.1):
        """
        weight function - as dist gets smaller, so does denominator and weight grows
        """
        return num / (dist + const)

    def subtract_wgt(self, dist, const=1.0):
        """
        subtracts distance from a constant, so smaller dist, higher weight
        if dist is greater than const, won't return any weight
        """
        if dist > const:
            return 0
        return const - dist

    def gaussian_wgt(self, dist, sigma=5.0):
        """
        weight is 1 when distance is 0
        weight never falls all the way to 0
        """
        return math.e**(-dist**2 / (2 * sigma**2))

    def predict(self, vectors):
        """
        Predicts the outcome of a list of entries
        """
        ests = []
        for vec in vectors:
            ests.append(self.knn_func(vec))
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
            guesses = self.predict(self.test)
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





# Returns the probability that the price is in the range specified based on the nearby nodes
# def probguess(data,vec1,low,high,k=5,weightf=gaussian_wgt):
#   dlist=getdistances(data,vec1)
#   nweight=0.0
#   tweight=0.0

#   for i in range(k):
#     dist=dlist[i][0]
#     idx=dlist[i][1]
#     weight=weightf(dist)
#     v=data[idx]['result']

#     # Is this point in the range?
#     if v>=low and v<=high:
#       nweight+=weight
#     tweight+=weight
#   if tweight==0: return 0

#   # The probability is the weights in the range divided by all the weights
#   return nweight/tweight


# def cumulativegraph(data,vec1,high,k=5,weightf=gaussian_wgt):
#   t1=arange(0.0,high,0.1)
#   cprob=array([probguess(data,vec1,0,v,k,weightf) for v in t1])
#   plt.plot(t1,cprob)
#   plt.ylim(0,1)
#   plt.savefig('/home/ubuntu/workspace/collective_intelligence/8ch/cum_density.jpg')
#   plt.close()


# def probabilitygraph(data,vec1,high,k=5,weightf=gaussian_wgt,ss=5.0):
#   # Make a range for the prices
#   t1=arange(0.0,high,0.1)

#   # Get the probabilities for the entire range
#   probs=[probguess(data,vec1,v,v+0.1,k,weightf) for v in t1]

#   # Smooth them by adding the gaussian of the nearby probabilites
#   smoothed=[]
#   for i in range(len(probs)):
#     sv=0.0
#     for j in range(0,len(probs)):
#       # farther away points have greater dist which mutes their weight
#       dist=abs(i-j)*0.1
#       weight=gaussian(dist,sigma=ss)
#       sv+=weight*probs[j]
#     smoothed.append(sv)
#   smoothed=array(smoothed)

#   plot(t1,smoothed)
#   plt.savefig('/home/ubuntu/workspace/collective_intelligence/8ch/prob_density.jpg')
#   plt.close()


if __name__ == '__main__':
    # regression based data
    # data = wineset1()
    DATA = wineset2()
    KNN_INST = KNN(np.array(DATA), opt=True)
    print(KNN_INST.run(10))

    # categorical data
    pdb.set_trace()
    IRIS = datasets.load_iris()
    IRIS = pd.merge(pd.DataFrame(IRIS.data[:, :2]), pd.DataFrame(IRIS.target),
                    left_index=True, right_index=True)

    KNN_INST = KNN(IRIS.values, cat=True)
    print(KNN_INST.run(10))

    plot_decision_regions(KNN_INST.train[:, :-1], KNN_INST.train[:, -1],
                          classifier=KNN_INST)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.savefig(IMG_ROOT + "knn_custom_{}.png"
                           "".format(dt.datetime.now().strftime("%Y%m%d")))
    plt.close()

    # Optimization
    # costf = createcostfunction(knnestimate,data)
    # print(optimization.annealingoptimize(weightdomain, costf, step=2))
    # print(optimization.geneticoptimize(weightdomain, costf, popsize=5, elite=0.2, maxiter=20))

    # Uneven Distributions
    # print(wineprice(99.0, 20.0))
    # print(weightedknn(data, [99.0, 20.0]))
    # print(crossvalidate(weightedknn, data))

    # Probability Density
    # print(probguess(data, [99, 20], 40, 80))
    # print(probguess(data, [99, 20], 80, 120))
    # print(probguess(data, [99, 20], 120, 1000))
    # print(probguess(data, [99, 20], 30, 120))

    # Graphing Density
    # cumulativegraph(data, (1,1), 120)
    # probabilitygraph(data, (99,20), 120)
