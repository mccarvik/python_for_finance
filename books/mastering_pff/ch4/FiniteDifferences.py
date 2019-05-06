""" Shared attributes and functions of FD """
import numpy as np

# Wikipedia describing the method
# https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#The_Black%E2%80%93Scholes_equation

# Applies the Black-Scholes Partial Differential Equation (PDE) framework
# Uses the partial derivatives to give a theoretical estimate of the option price

class FiniteDifferences(object):

    def __init__(self, S0, K, r, T, sigma, Smax, M, N,
                 is_call=True):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.Smax = Smax
        self.M, self.N = int(M), int(N)  # Ensure M&N are integers
        self.is_call = is_call

        self.dS = Smax / float(self.M)
        self.dt = T / float(self.N)
        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        # Create M x N grid
        self.grid = np.zeros(shape=(self.M+1, self.N+1))
        # set up boundary conditions at the extreme ends of the nodes
        self.boundary_conds = np.linspace(0, Smax, self.M+1)

    def _setup_boundary_conditions_(self):
        pass

    def _setup_coefficients_(self):
        pass

    def _traverse_grid_(self):
        """  Iterate the grid backwards in time """
        pass

    def _interpolate_(self):
        """
        Use piecewise linear interpolation on the initial
        grid column to get the closest price at S0.
        """
        return np.interp(self.S0,
                         self.boundary_conds,
                         self.grid[:, 0])

    def price(self):
        self._setup_boundary_conditions_()
        self._setup_coefficients_()
        self._traverse_grid_()
        return self._interpolate_()