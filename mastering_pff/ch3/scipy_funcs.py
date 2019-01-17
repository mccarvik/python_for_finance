"""
Documentation at
http://docs.scipy.org/doc/scipy/reference/optimize.html
"""
import scipy.optimize as optimize

y = lambda x: x**3 + 2.*x**2 - 5.
dy = lambda x: 3.*x**2 + 4.*x

# Compare each function provided by scipy

# Call method: bisect(f, a, b[, args, xtol, rtol, maxiter, ...])
print("Bisection method: %s" % optimize.bisect(y, -5., 5., xtol=0.00001))

# Call method: newton(func, x0[, fprime, args, tol, ...])
print("Newton's method: %s" % optimize.newton(y, 5., fprime=dy))
# When fprime=None, then the secant method is used.
print("Secant method: %s" % optimize.newton(y, 5.))

# Brent method combines the previous methods based on the inputs and function
# Call method: brentq(f, a, b[, args, xtol, rtol, maxiter, ...])
print("Brent's method: %s" % optimize.brentq(y, -5., 5.))