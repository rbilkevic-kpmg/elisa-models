import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from utils import residuals_func


def logistic_4(x, a, b, c, d):
    """
    4PL logistic equation
    :param nd_array x: signal value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :return: nd_array
    """
    return (a - d) / (1 + (x / c) ** b) + d


def inv_logistic_4(y, a, b, c, d):
    """
    Inverse 4PL logistic equation
    :param nd_array y: response value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :return: nd_array
    """
    return c * ((a - y) / (y - d)) ** (1 / b)


# Data
x_graph = np.linspace(80, 0.1, 100)
x = np.array([60, 30, 15, 7.5, 3.75, 1.875, 0.9375])
#a, b, c, d = 0.5, 2.5, 8, 9.1
#y_true = logistic_4(x, a, b, c, d)
y_meas = np.array([0.4295, 0.6265, 0.9585, 1.2785, 1.6825,  1.8275, 2.102])

# Initial set of parameters
p_init = np.array([2, 4, 9, 5])

# Fit equation using least squares
p_optim = leastsq(residuals_func(logistic_4), p_init, args=(y_meas, x))
print(p_optim[0])

# Plot results
plt.plot(x_graph, logistic_4(x_graph, *p_optim[0]), x, y_meas, 'o')
#plt.scatter(inv_logistic_4(np.array([1.07]), *p_optim[0]) ,np.array([1.07]), marker='+', c='red')
plt.legend(['Fit', 'Measured', 'Model'])
plt.xlabel('Concentration')
plt.ylabel('Density')
#for i, (param, actual, est) in enumerate(zip('ABCF'), [a, b, c, d], p_optim[0])
plt.show()


#print(inv_logistic_4(np.array([1.07]), *p_optim[0]))
