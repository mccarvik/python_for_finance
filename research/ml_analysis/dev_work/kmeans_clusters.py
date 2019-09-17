"""
Module to calculate kmeans clusters
"""
import pdb
import random
from math import sqrt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.ml_utils import IMG_ROOT
from knn import euclidean
mpl.use('Agg')

PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/png/temp/"

def read_blog_file(filename=None):
    """
    Reads the blog info from the file
    """
    if not filename:
        filename = "/home/ec2-user/environment/python_for_finance/research/" \
                   "ml_analysis/dev_work/dev_data/blogdata1.txt"
    with open(filename) as file:
        lines = file.readlines()

    # First line is the column titles
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        line_strip = line.strip().split('\t')
        # First column in each row is the rowname
        rownames.append(line_strip[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in line_strip[1:]])
    return (rownames, colnames, data)


def pearson_corr(vec1, vec2):
    """
    Calculate the pearson correlation coefficient between 2 vectors (aka r)
    A measure of the linear correlation between two variables
    """
    # Simple sums
    sum1 = sum(vec1)
    sum2 = sum(vec2)

    # Sums of the squares
    sum1_sq = sum([v**2 for v in vec1])
    sum2_sq = sum([v**2 for v in vec2])

    # Sum of the products
    p_sum = sum([vec1[i] * vec2[i] for i in range(len(vec1))])

    # Calculate r (Pearson score)
    # Covariance
    num = (p_sum) - ((sum1 * sum2) / len(vec1))
    # stdev X * stdev Y
    den = sqrt((sum1_sq - sum1**2 / len(vec1)) * (sum2_sq - sum2**2 / len(vec1)))
    if den == 0:
        return 0
    # NOTE: will return 1 - r so lower number = higher distance for values
    # that are farther apart
    return 1.0 - (num / den)


class Bicluster:
    """
    Class to create cluster instances
    """
    def __init__(self, vec, left=None, right=None, distance=0.0, clust_id=None):
        """
        Constructor function for bicluster
        """
        self.left = left
        self.right = right
        self.vec = vec
        self.clust_id = clust_id
        self.distance = distance


def hcluster(rows, distance=pearson_corr):
    """
    Function for hierarchical clustering
    """
    distances = {}
    currentclustid = -1

    # Clusters are initially just the original items
    clust = [Bicluster(rows[i], clust_id=i) for i in range(len(rows))]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i, _ in enumerate(clust):
            for j in range(i + 1, len(clust)):
                # check if we already have this distance
                if (clust[i].clust_id, clust[j].clust_id) not in distances:
                    distances[(clust[i].clust_id, clust[j].clust_id)] = distance(clust[i].vec, \
                                                                        clust[j].vec)
                ind_d = distances[(clust[i].clust_id, clust[j].clust_id)]

                # save the closest distance
                if ind_d < closest:
                    closest = ind_d
                    lowestpair = (i, j)

        # lowestpair has id of two closests
        # Get the average point of every value in the two clusters' vectors
        merge_vec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])
                     / 2.0 for i in range(len(clust[0].vec))]

        # create the new cluster
        newcluster = Bicluster(merge_vec, left=clust[lowestpair[0]],
                               right=clust[lowestpair[1]], distance=closest,
                               clust_id=currentclustid)

        # all new clusters will have negative ids
        currentclustid -= 1
        # add new cluster to the set and delete the two old ones
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    return clust[0]


def print_clust(clust, labels=None, indnt=0):
    """
    Print the cluster breakdown
    """
    # indent to make a hierarchy layout
    for _ in range(indnt):
        print(' ', end='')
    if clust.clust_id < 0:
        # negative id means that this is branch
        print('-')
    else:
        # positive id means that this is an endpoint
        if not labels:
            print(clust.clust_id)
        else:
            print(labels[clust.clust_id])

    # now print the right and left branches
    if clust.left != None:
        print_clust(clust.left, labels=labels, indnt=indnt + 1)
    if clust.right != None:
        print_clust(clust.right, labels=labels, indnt=indnt + 1)


def get_height(clust):
    """
    Get the height of the cluster
    """
    # if endpoint then the height is just 1
    if clust.left is None and clust.right is None:
        return 1
    # if not end point, height is the same of the heights of each branch
    return get_height(clust.left) + get_height(clust.right)


def get_depth(clust):
    """
    Gets the total error of the node
    which is the maximum of the tow branches + its own dist
    """
    # The distance of an endpoint is 0.0
    if clust.left is None and clust.right is None:
        return 0

    # The dist of a branch is the greater of its two sides plus its own dist
    return max(get_depth(clust.left), get_depth(clust.right)) + clust.distance


def draw_dendrogram(clust, labels, jpeg='clusters.jpg'):
    """
    Draws the dendogram
    """
    # height and width
    height = get_height(clust) * 20
    width = 1200
    depth = get_depth(clust)

    # width is fixed, so scale distances accordingly
    scaling = float(width - 150) / depth

    # Create a new image with a white background
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((0, height / 2, 10, height / 2), fill=(255, 0, 0))

    # Draw the first node
    draw_node(draw, clust, 10, height / 2, scaling, labels)
    img.save(PATH + jpeg, 'JPEG')


def draw_node(draw, clust, x_val, y_val, scaling, labels):
    """
    Draws an individual node on the image
    """
    if clust.clust_id < 0:
        hgt1 = get_height(clust.left) * 20
        hgt2 = get_height(clust.right) * 20
        top = y_val - (hgt1 + hgt2) / 2
        bottom = y_val + (hgt1 + hgt2) / 2
        # Line length
        line_l = clust.distance * scaling
        # Vertical line from this cluster to children
        draw.line((x_val, top + hgt1 / 2, x_val, bottom - hgt2 / 2),
                  fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x_val, top + hgt1 / 2, x_val + line_l, top + hgt1 / 2),
                  fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x_val, bottom - hgt2 / 2, x_val + line_l, bottom - hgt2 / 2),
                  fill=(255, 0, 0))

        # Call the function to draw the left and right nodes
        draw_node(draw, clust.left, x_val + line_l, top + hgt1 / 2, scaling, labels)
        draw_node(draw, clust.right, x_val + line_l, bottom - hgt2 / 2, scaling, labels)
    else:
        # If this is an endpoint, draw the item label
        try:
            draw.text((x_val + 5, y_val - 7), labels[clust.clust_id], (0, 0, 0))
        except Exception as exc:
            print("Probably a unicode symbol issue" + str(exc))


def rotate_matrix(data):
    """
    rotates the matrix
    """
    newdata = []
    for i, _ in enumerate(data[0]):
        newrow = [data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata


def kcluster(rows, distance=pearson_corr, k=4):
    """
    Algorithm for developing the location of the k clusters for the dataset
    """
    # Determine the minimum and maximum values of each vector
    pdb.set_trace()
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
              for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))] for j in range(k)]
    print(clusters)
    lastmatches = None
    iterations = 3
    # Go through x number of iterations to place centroids
    for _ in range(iterations):
        print('Iteration %d' % _)
        bestmatches = [[] for i in range(k)]

        # Find which centroid is the closest for each row
        for j, _ in enumerate(rows):
            row = rows[j]
            bestmatch = 0
            # find the closest centroid for this row
            for i in range(k):
                ind_dist = distance(clusters[i], row)
                if ind_dist < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, algo is done
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # Move the centroids to the average of their members
        # pdb.set_trace()
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if bestmatches[i]:
                # only grab rows belonging to this centroid
                for rowid in bestmatches[i]:
                    for val, _ in enumerate(rows[rowid]):
                        # add each val in each vector to the avgs
                        avgs[val] += rows[rowid][val]
                for j, _ in enumerate(avgs):
                    # divide by number of vectors to get the average
                    avgs[j] /= len(bestmatches[i])
                # this becomes the location of the centroid
                clusters[i] = avgs
        print(clusters)
    return (np.array(bestmatches), np.array(clusters))


def tanamoto(vec1, vec2):
    """
    ratio of the intersection set to the union set
    """
    (class1, class2, shr) = (0, 0, 0)
    for i in enumerate(vec1):
        # in vector 1
        if vec1[i] != 0:
            class1 += 1
        # in vector 2
        if vec2[i] != 0:
            class2 += 1
        # in both
        if vec1[i] != 0 and vec2[i] != 0:
            shr += 1
    return 1.0 - float(shr) / (class1 + class2 - shr)


def scaledown(data, distance=pearson_corr, rate=0.01):
    """
    Tries to find a two dimentional representation of dataset
    Tries to create a chart where the distances between items match their
    multidimenstional distance
    """
    num = len(data)

    # The real distances between every pair of items
    realdist = [[distance(data[i], data[j]) for j in range(num)] for i in
                range(0, num)]

    # Randomly initialize the starting points of the locations in 2D
    loc = [[random.random(), random.random()] for i in range(num)]
    fakedist = [[0.0 for j in range(num)] for i in range(num)]

    lasterror = None
    for _ in range(0, 1000):
        # Find projected distances
        for i in range(num):
            for j in range(num):
                fakedist[i][j] = sqrt(sum([pow(loc[i][x] - loc[j][x], 2)
                                           for x in range(len(loc[i]))]))

        # Move points
        grad = [[0.0, 0.0] for i in range(num)]

        totalerror = 0
        for k in range(num):
            for j in range(num):
                if j == k:
                    continue
                # Get the error % difference between the real and est dists
                errorterm = (fakedist[j][k] - realdist[j][k]) / realdist[j][k]

                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                grad[k][0] += (loc[k][0] - loc[j][0]) / fakedist[j][k] * errorterm
                grad[k][1] += (loc[k][1] - loc[j][1]) / fakedist[j][k] * errorterm

                # Keep track of the total error
                totalerror += abs(errorterm)
        print(totalerror)

        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror < totalerror:
            break
        lasterror = totalerror

        # Move each of the points by the learning rate times the gradient
        for k in range(num):
            loc[k][0] -= rate * grad[k][0]
            loc[k][1] -= rate * grad[k][1]
    return loc


def draw2d(data, labels, jpeg='mds2d.jpg'):
    """
    draw the 2 dimensional description of multidimensonal data
    """
    img = Image.new('RGB', (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, _ in enumerate(data):
        x_val = (data[i][0] + 0.5) * 1000
        y_val = (data[i][1] + 0.5) * 1000
        draw.text((x_val, y_val), labels[i], (0, 0, 0))
    img.save(PATH + jpeg, 'JPEG')


def test_kmeans():
    """
    test custom algo vs sklearn
    """
    clusts = 3
    # Dont need y_vals, unsupervised
    x_vals, _ = make_blobs(n_samples=150, n_features=2, centers=clusts,
                           cluster_std=0.5, shuffle=True, random_state=0)
    plt.scatter(x_vals[:, 0], x_vals[:, 1], c='white', marker='o', s=50)
    plt.grid()
    plt.tight_layout()

    # Sklearn solution
    kmeans = KMeans(n_clusters=clusts, init='random', n_init=10, max_iter=300,
                    tol=1e-04, random_state=0)
    y_km = kmeans.fit_predict(x_vals)

    # Custom solution
    k_clust = kcluster(x_vals, k=clusts, distance=euclidean)
    # print([BLOGNAMES[r] for r in KCLUST[0]])

    # clusters
    plt.scatter(x_vals[y_km == 0, 0], x_vals[y_km == 0, 1], s=50, c='lightgreen',
                marker='s', label='cluster 1')
    plt.scatter(x_vals[y_km == 1, 0], x_vals[y_km == 1, 1], s=50, c='orange',
                marker='o', label='cluster 2')
    plt.scatter(x_vals[y_km == 2, 0], x_vals[y_km == 2, 1], s=50, c='lightblue',
                marker='v', label='cluster 3')
    # sklearn centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250,
                marker='*', c='red', label='centroids')
    # custom kmeans clusters centroids
    plt.scatter(k_clust[1][:, 0], k_clust[1][:, 1], s=250, marker='X', c='purple',
                label='centroids')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(IMG_ROOT + "PML/" + 'kmeans_comp.png', dpi=300)
    plt.close()
    print('Distortion: %.2f' % kmeans.inertia_)


if __name__ == "__main__":
    test_kmeans()
    # BLOGNAMES, WORDS, DATA = read_blog_file()

    # Hierarchical clustering
    # CLUST = hcluster(DATA)
    # print_clust(CLUST, labels=BLOGNAMES)
    # draw_dendrogram(CLUST, BLOGNAMES, jpeg='blogclust.jpg')

    # Kmeans Clustering
    # KCLUST = kcluster(DATA, k=10)
    # print([BLOGNAMES[r] for r in KCLUST[0]])


    ##### Two-D display example #####
    # COORDS = scaledown(DATA)
    # draw2d(COORDS, BLOGNAMES, jpeg='blogs2d.jpg')
