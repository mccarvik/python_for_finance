"""
Module to implement Support Vector Machines
"""
import pdb
from libsvm import svm


def demo1():
    """
    Set of exercises to better understand workings of SVM
    """
    pdb.set_trace()
    prob = svm.svm_problem([1, -1], [[1, 0, 1], [-1, 0, -1]])
    param = svm.svm_parameter()
    mod = svm.svm_model(prob, param)


if __name__ == '__main__':
    demo1()
