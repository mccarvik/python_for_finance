"""
Module for Advanced Classification and Kernel methods
"""
import pdb
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

DATA_PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/dev_work/dev_data/"
PNG_PATH = "/home/ec2-user/environment/python_for_finance/research/ml_analysis/png/temp/"

class MatchRow:
    """
    Hold match data
    """
    def __init__(self, row, allnum=False):
        """
        Constructor
        """
        if allnum:
            self.data = [float(row[i]) for i in range(len(row)-1)]
        else:
            self.data = row[0:len(row) - 1]
        self.match = int(row[len(row) - 1])


def load_match(file, allnum=False):
    """
    read the match data from file
    """
    rows = []
    with open(DATA_PATH + file) as read_f:
        lines = read_f.readlines()
    for line in lines:
        rows.append(MatchRow(line.split(','), allnum))
    return rows


def plot_age_matches(rows):
    """
    Plots the age matches in the data
    """
    xdm, ydm = [r.data[0] for r in rows if r.match == 1], \
               [r.data[1] for r in rows if r.match == 1]
    xdn, ydn = [r.data[0] for r in rows if r.match == 0], \
               [r.data[1] for r in rows if r.match == 0]

    plt.plot(xdm, ydm, 'bo')
    plt.plot(xdn, ydn, 'b+')
    plt.savefig(PNG_PATH + "age_matches.jpg")
    plt.close()


def linear_train(rows):
    """
    Finding the average of each class of data
    Then seeing which average a data point is closer to
    """
    averages = {}
    counts = {}

    for row in rows:
        # Get the class of this point
        ind_cl = row.match

        averages.setdefault(ind_cl, [0.0] * (len(row.data)))
        counts.setdefault(ind_cl, 0)

        for i, _ in enumerate(row.data):
            # Add each point on this vector to the averages
            averages[ind_cl][i] += float(row.data[i])

        # Keep track of how many points in each class
        counts[ind_cl] += 1

    # Divide sums by counts to get the averages
    for ind_cl, avg in averages.items():
        for i, _ in enumerate(avg):
            avg[i] /= counts[ind_cl]
    return averages


def dot_prod(vec1, vec2):
    """
    Dot product of two vectors
    """
    return sum([vec1[i] * vec2[i] for i, _ in enumerate(vec1)])


def veclength(vec):
    """
    Gets the length of a vector
    """
    return sum([p**2 for p in vec])


def dpclassify(point, avgs):
    """
    Dot product will identify the angle between point to be classified and
    vector between the 2 averages
    Angle will dictate what side of the line separating the averages the point
    falls on
    class = sign(X dot M0 - X dot M1) + ((M0 dot M0 - M1 dot M1) / 2)
    """
    # (M0 dot M0 - M1 dot M1) / 2
    b_val = (dot_prod(avgs[1], avgs[1]) - dot_prod(avgs[0], avgs[0])) / 2
    # (X dot M0 - X dot M1) + b
    y_val = dot_prod(point, avgs[0]) - dot_prod(point, avgs[1]) + b_val
    # sign
    if y_val > 0:
        return 0
    return 1


def yesno(val):
    """
    Convert categorical data to numerical
    """
    if val == 'yes':
        return 1
    elif val == 'no':
        return -1
    return 0


def matchcount(interest1, interest2):
    """
    Count the number of common interests
    """
    int1 = interest1.split(':')
    int2 = interest2.split(':')
    tot = 0
    for val in int1:
        if val in int2:
            tot += 1
    return tot


def load_numerical():
    """
    Load the numerical data from match maker dataset
    """
    oldrows = load_match('matchmaker.csv')
    newrows = []
    for row in oldrows:
        dat = row.data
        # data=[float(dat[0]), yesno(dat[1]), yesno(dat[2]), float(dat[5]), yesno(dat[6]),
        #       yesno(dat[7]), matchcount(dat[3], dat[8]), milesdistance(d[4], dat[9]),
        #       row.match]

        # yahoo api doesnt work for distance
        data = [float(dat[0]), yesno(dat[1]), yesno(dat[2]), float(dat[5]),
                yesno(dat[6]), yesno(dat[7]), matchcount(dat[3], dat[8]), 0, row.match]
        newrows.append(MatchRow(data))
    return newrows


def scale_data(rows):
    """
    Need to scale the data as the values may differ in magnitude
    """
    low = [999999999.0] * len(rows[0].data)
    high = [-999999999.0] * len(rows[0].data)
    # Find the lowest and highest values
    for row in rows:
        dat = row.data
        for i, _ in enumerate(dat):
            if dat[i] < low[i]:
                low[i] = dat[i]
            if dat[i] > high[i]:
                high[i] = dat[i]

    # Create a function that scales data
    def scaleinput(dat):
        """
        Scales the input for this set
        """
        return [(dat[i] - low[i]) / (high[i] - low[i]) for i in range(len(low)-1)]

    # Scale all the data
    newrows = [MatchRow(scaleinput(row.data) + [row.match]) for row in rows]

    # Return the new data and the function
    return newrows, scaleinput


def rbf(vec1, vec2, gamma=10):
    """
    Radial Basis Function
    Used to transform data from more complex spaces
    Returns what the dot-product would have been if the data had first been
    transformed to a higher dimensional space using some mapping function
    """
    # take the differences of the vectors
    d_v = [vec1[i] - vec2[i] for i, _ in enumerate(vec1)]
    # get the length
    length = veclength(d_v)
    # apply radial basis functon
    return math.e**(-gamma * length)


def nlclassify(point, rows, offset, gamma=10):
    """
    nclassify a datapoint based on rbf radial transformation
    """
    sum0 = 0.0
    sum1 = 0.0
    count0 = 0
    count1 = 0
    for row in rows:
        if row.match == 0:
            sum0 += rbf(point, row.data, gamma)
            count0 += 1
        else:
            sum1 += rbf(point, row.data, gamma)
            count1 += 1

    # exact same process as the dotproduct (dpclassify) but using different function (rbf)
    y_val = (1.0 / count0) * sum0 - (1.0 / count1) * sum1 + offset

    if y_val > 0:
        return 0
    return 1


def get_offset(rows, gamma=10):
    """
    Functions as a pseduo "y-intercept" for the rbf radial calculation
    """
    len0 = []
    len1 = []
    # divide up the classes between matches and not matches
    for row in rows:
        if row.match == 0:
            len0.append(row.data)
        else:
            len1.append(row.data)

    # Calculates the sum of the dot product of every vector with evey other
    # vector in the class
    # Takes a while to run as the values arent averaged like for the
    # dotproduct offset calc so have to run thru each row
    sum0 = sum(sum([rbf(v1, v2, gamma) for v1 in len0]) for v2 in len0)
    sum1 = sum(sum([rbf(v1, v2, gamma) for v1 in len1]) for v2 in len1)

    # pseudo averaging process of the sums
    return (1.0 / (len(len1)**2)) * sum1 - (1.0 / (len(len0)**2)) * sum0


if __name__ == "__main__":
    AGES_ONLY = load_match('agesonly.csv', allnum=True)
    MATCH_MAKER = load_match('matchmaker.csv')

    # plot ages matches
    # plot_age_matches(AGES_ONLY)

    # Linear Classifier
    # AVGS = linear_train(AGES_ONLY)
    # print(dpclassify([30, 30], AVGS))
    # print(dpclassify([25, 40], AVGS))

    # Scaling the data
    NUM_EST = load_numerical()
    SCALEDSET, SCALEF = scale_data(NUM_EST)
    # AVGS = linear_train(SCALEDSET)

    # Kernel Trick - nonlinear data
    # OFFSET = get_offset(AGES_ONLY)
    # print(nlclassify([30, 30], AGES_ONLY, OFFSET))
    # print(nlclassify([30, 25], AGES_ONLY, OFFSET))
    # print(nlclassify([25, 40], AGES_ONLY, OFFSET))
    # print(nlclassify([48, 20], AGES_ONLY, OFFSET))

    # Using all the data
    SSOFFSET = get_offset(SCALEDSET)
    NEW_ROW_1 = [28.0, -1, -1, 26.0, -1, 1, 2, 0.8]  # Man doesn't want children, woman does
    print(nlclassify(SCALEF(NEW_ROW_1), SCALEDSET, SSOFFSET))
    NEW_ROW_2 = [28.0, -1, 1, 26.0, -1, 1, 2, 0.8]  # Both want children
    print(nlclassify(SCALEF(NEW_ROW_2), SCALEDSET, SSOFFSET))
    