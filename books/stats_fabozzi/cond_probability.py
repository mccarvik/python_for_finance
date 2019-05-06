import sys, pdb
sys.path.append("/home/ubuntu/workspace/python_for_finance")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
sys.path.append("/usr/local/lib/python3.4/dist-packages")
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt, pi, log, e

# from dx import *
# from utils.utils import *

import numpy as np
import pandas as pd
import datetime as dt


# NOTES
# P(A int B) = P(A|B)P(B)
# P(A int B) = P(B|A)P(A)
# P(A) = P(A int B) + P(A int Bc)
# P(A) = P(A|B)P(B) + P(A|Bc)P(Bc)      Law of Total Probability
# Conditional Variance = the variance calculated on the new subspace
# ex: Var(A|B) calculates the variance of A using only the data points where B occurred
# Expected tail loss --> expected loss when VaR limit is breached. i.e. just calculating the conditional expected return
# aka CVaR --> conditional Value at Risk



def uncond_prob(probs, item_ind):
    p = probs[item_ind]
    return p / sum(probs)

    
def cond_prob(target_event, given_event, probs):
    # Prob of A given B is the intersection of A and B divided by the probability of B
    # P(A|B) = P(A int B) / P(B)
    # given_event represents B, is a list
    inter = [val for val in target_event if val in given_event]
    inter_P = sum([probs[x] for x in inter]) / sum(probs)
    upd_probs = [probs[x] for x in given_event]
    PB = sum(upd_probs) / sum(probs)
    return inter_P / PB


def indep_check(probs_A, probs_B, probs):
    # P(A)P(B) == P(A int B) if A and B are independent
    PA = sum([probs[x] for x in probs_A]) / sum(probs)
    PB = sum([probs[x] for x in probs_B]) / sum(probs)
    inter = [val for val in probs_A if val in probs_B]
    inter_P = sum([probs[x] for x in inter]) / sum(probs)
    return PA * PB == inter_P


def bayes_rule(PA_givB, PA, PB):
    # returns PB_givA
    return (PA_givB * PB) / PA


if __name__ == '__main__':
    # print(uncond_prob([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 1))
    # print(cond_prob([1], [1, 3, 5], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
    # print(indep_check([1], [1, 3, 5], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]))
    print(bayes_rule(0.75,0.51,0.4))
    
    