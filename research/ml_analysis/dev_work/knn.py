from random import random, randint
import math, pdb, sys
sys.path.append("/home/ubuntu/workspace/collective_intelligence")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# from ch5 import optimization
from pylab import *


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
    # Creates a bunch of reasonable prices with a little randomness thrown into the price
    """
    rows=[]
    for i in range(300):
        # Create a random age and rating
        rating=random()*50+50
        age=random()*50
    
        # Get reference price
        price=wineprice(rating,age)
        
        # Add some noise
        price*=(random()*0.2+0.9)
    
        # Add to the dataset
        rows.append({'input':(rating,age),
                     'result':price})
    return rows


def euclidean(v1,v2):
    """
    # Pythagorean Theorem for distances
    """
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i])**2
    return math.sqrt(d)


def get_distances(data, vec1):
    """
    Calcs the distance from a the given vector compared with every other vector
    Returns a sorted list with the closest vectors at the top
    """
    distancelist = []
    # Loop over every item in the dataset
    for i in range(len(data)):
        vec2 = data[i]['input']
    
    # Add the distance and the index
    distancelist.append((euclidean(vec1, vec2), i))
    # Sort by distance
    distancelist.sort()
    return distancelist


def knnestimate(data,vec1,k=5):
  # Get sorted distances
  dlist=getdistances(data,vec1)
  avg=0.0
  
  # Take the average of the top k results
  for i in range(k):
    idx=dlist[i][1]
    avg+=data[idx]['result']
  avg=avg/k
  return avg


def inverseweight(dist,num=1.0,const=0.1):
  return num/(dist+const)


def subtractweight(dist,const=1.0):
  if dist>const: 
    return 0
  else: 
    return const-dist


def gaussian(dist,sigma=5.0):
  return math.e**(-dist**2/(2*sigma**2))


def weightedknn(data,vec1,k=5,weightf=gaussian):
  # Get distances
  dlist=getdistances(data,vec1)
  avg=0.0
  totalweight=0.0
  
  # Get weighted average
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    # Adds up all the results and multiplies more for closer data points, than averages over all the k nodes
    avg+=weight*data[idx]['result']
    totalweight+=weight
  if totalweight==0: return 0
  avg=avg/totalweight
  return avg


def dividedata(data,test=0.05):
  trainset=[]
  testset=[]
  for row in data:
    if random()<test:
      testset.append(row)
    else:
      trainset.append(row)
  return trainset,testset


def testalgorithm(algf,trainset,testset):
  error=0.0
  for row in testset:
    guess=algf(trainset,row['input'])
    error+=(row['result']-guess)**2
  return error/len(testset)


def crossvalidate(algf,data,trials=100,test=0.1):
  error=0.0
  for i in range(trials):
    trainset,testset=dividedata(data,test)
    error+=testalgorithm(algf,trainset,testset)
  return error/trials


def wineset2():
  rows=[]
  for i in range(300):
    rating=random()*50+50
    age=random()*50
    aisle=float(randint(1,20))
    bottlesize=[375.0,750.0,1500.0][randint(0,2)]
    price=wineprice(rating,age)
    price*=(bottlesize/750)
    price*=(random()*0.2+0.9)
    rows.append({'input':(rating,age,aisle,bottlesize),
                 'result':price})
  return rows


def rescale(data,scale):
  scaleddata=[]
  for row in data:
    scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
    scaleddata.append({'input':scaled,'result':row['result']})
  return scaleddata


def createcostfunction(algf,data):
  def costf(scale):
    sdata=rescale(data,scale)
    return crossvalidate(algf,sdata,trials=20)
  return costf


weightdomain=[(0,10)]*4


def wineset3():
  rows=wineset1()
  for row in rows:
    if random()<0.5:
      # Wine was bought at a discount store
      row['result']*=0.6
  return rows


# Returns the probability that the price is in the range specified based on the nearby nodes
def probguess(data,vec1,low,high,k=5,weightf=gaussian):
  dlist=getdistances(data,vec1)
  nweight=0.0
  tweight=0.0
  
  for i in range(k):
    dist=dlist[i][0]
    idx=dlist[i][1]
    weight=weightf(dist)
    v=data[idx]['result']
    
    # Is this point in the range?
    if v>=low and v<=high:
      nweight+=weight
    tweight+=weight
  if tweight==0: return 0
  
  # The probability is the weights in the range divided by all the weights
  return nweight/tweight


def cumulativegraph(data,vec1,high,k=5,weightf=gaussian):
  t1=arange(0.0,high,0.1)
  cprob=array([probguess(data,vec1,0,v,k,weightf) for v in t1])
  plt.plot(t1,cprob)
  plt.ylim(0,1)
  plt.savefig('/home/ubuntu/workspace/collective_intelligence/8ch/cum_density.jpg')
  plt.close()


def probabilitygraph(data,vec1,high,k=5,weightf=gaussian,ss=5.0):
  # Make a range for the prices
  t1=arange(0.0,high,0.1)
  
  # Get the probabilities for the entire range
  probs=[probguess(data,vec1,v,v+0.1,k,weightf) for v in t1]
  
  # Smooth them by adding the gaussian of the nearby probabilites
  smoothed=[]
  for i in range(len(probs)):
    sv=0.0
    for j in range(0,len(probs)):
      # farther away points have greater dist which mutes their weight
      dist=abs(i-j)*0.1
      weight=gaussian(dist,sigma=ss)
      sv+=weight*probs[j]
    smoothed.append(sv)
  smoothed=array(smoothed)
    
  plot(t1,smoothed)
  plt.savefig('/home/ubuntu/workspace/collective_intelligence/8ch/prob_density.jpg')
  plt.close()
  

if __name__ == '__main__':
    data = wineset1()

    # Original testing
    print(wineprice(95.0, 3.0))
    print(wineprice(95.0, 8.0))
    
    # Defining Similarity
    # print(data[0]['input'])
    # print(data[1]['input'])
    # print(knnestimate(data, (95.0, 3.0)))
    # print(knnestimate(data, (99.0, 3.0)))
    # print(knnestimate(data, (99.0, 5.0)))
    # print(wineprice(99, 5.0))
    # print(knnestimate(data, (99.0, 5.0), k=1))
    
    # Weighted Neighbors
    # print(subtractweight(0.1))
    # print(inverseweight(0.1))
    # print(gaussian(0.1))
    # print(gaussian(1))
    # print(subtractweight(1.0))
    # print(inverseweight(1))
    # print(gaussian(3))
    # print(weightedknn(data, (99.0, 5.0)))
    
    
    # Cross Validation
    # print(crossvalidate(knnestimate, data))
    # knn3 = lambda d, v: knnestimate(d, v, k=3)
    # print(crossvalidate(knn3, data))
    # knn1 = lambda d, v: knnestimate(d, v, k=1)
    # print(crossvalidate(knn1, data))
    # print(crossvalidate(weightedknn, data))
    # knninverse = lambda d, v: weightedknn(d, v, weightf=inverseweight)
    # print(crossvalidate(knninverse, data))
    
    # Scaling Variables
    # knn3 = lambda d, v: knnestimate(d, v, k=3)
    # print(crossvalidate(knn3, data))
    # print(crossvalidate(weightedknn, data))
    
    # sdata = rescale(data, [10, 10, 0, 0.5])
    # print(crossvalidate(knn3, sdata))
    # print(crossvalidate(weightedknn, sdata))
    
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