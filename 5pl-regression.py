import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#plt.style.use('seaborn-paper')

from utils import (
    residuals_func,
    r_squared_adj
)
from data import (
    STANDARD_DATA,
    DATA
)


def logistic_5(x, a, b, c, d, e):
    """
    5PL logistic equation

    :param nd_array x: signal value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :param float e: asymmetry factor
    :return: nd_array
    """
    return (a - d) / (1 + (x / c) ** b) ** e + d


def inv_logistic_5(y, a, b, c, d, e):
    """
    Inverse 5PL logistic equation

    :param nd_array y: response value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :param float e: asymmetry factor
    :return: nd_array
    """
    return c * (((a - d) / (y - d)) ** (1 / e) - 1) ** (1 / b)


# Data
x = np.array([60, 30, 15, 7.5, 3.75, 1.875, 0.9375])
x_range = x.max() - x.min()
x_graph = np.linspace(x.min() - x_range * 0.25, x.max() + x_range * 0.25, 100)
# a, b, c, d, e = 0.5, 2.5, 8, 9.1, 14
# y_true = logistic_4(x, a, b, c, d)
y_meas = np.array(STANDARD_DATA)
y_range = y_meas.max() - y_meas.min()

# Initial set of parameters
p_init = np.array([2, 4, 9, 1, 4])

# Fit equation using least squares
p_optim = leastsq(residuals_func(logistic_5), p_init, args=(y_meas, x))
print(p_optim[0])
y_pred = logistic_5(x, *p_optim[0])
r_2 = r_squared_adj(y_meas, y_pred, len(x), len(p_optim))

# Plot results
plt.plot(x_graph, logistic_5(x_graph, *p_optim[0]), x, y_meas, 'o')
# plt.scatter(inv_logistic_5(np.array([1.07]), *p_optim[0]) ,np.array([1.07]), marker='+', c='red')
plt.legend(['Fit', 'Measured', 'Model'])
plt.title('5 Point Linear Regression')
plt.xlabel('8-OHdG Concentration, ng/ml')
plt.ylabel('Net Optical Density')

plt.text(x.min() + x_range * 0.05, y_meas.min() + y_range, r"$Adj. R^2={:.3f}$".format(r_2))
plt.show()

# print(inv_logistic_4(np.array([1.07]), *p_optim[0]))
