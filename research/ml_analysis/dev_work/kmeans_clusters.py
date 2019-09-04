"""
Module to calculate kmeans clusters
"""
import pdb
from math import sqrt
from PIL import Image, ImageDraw

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
                    distances[(clust[i].clust_id, clust[j].clust_id)] = distance(clust[i].vec, clust[j].vec)
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



if __name__ == "__main__":
    BLOGNAMES, WORDS, DATA = read_blog_file()
    CLUST = hcluster(DATA)
    print_clust(CLUST, labels=BLOGNAMES)
    draw_dendrogram(CLUST, BLOGNAMES, jpeg='blogclust.jpg')
