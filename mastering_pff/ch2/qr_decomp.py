""" QR decomposition with scipy """
import scipy
import numpy as np

A = np.array([
    [2., 1., 1.],
    [1., 3., 2.],
    [1., 0., 0]])
B = np.array([4., 5., 6.])

# Uses an orthogonal matrix to solve the system of equations
Q, R = np.linalg.qr(A)  # QR decomposition
y = np.dot(Q.T, B)  # Let y=Q`.B
x = np.linalg.solve(R, y)  # Solve Rx=y
print(x)