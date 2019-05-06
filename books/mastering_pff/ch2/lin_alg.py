""" Linear algebra with NumPy matrices """
import numpy as np

# Can use linear algebara to figure out how many positions of each stock to hold
# Given some constraints represented in the matrix
A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])
B = np.array([4, 5, 6])

print(np.linalg.solve(A, B))
