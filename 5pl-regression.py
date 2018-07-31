import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

from utils import (
    residuals_func,
    r_squared_adj
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
x_graph = np.linspace(80, 0.1, 100)
x = np.array([60, 30, 15, 7.5, 3.75, 1.875, 0.9375])
# a, b, c, d, e = 0.5, 2.5, 8, 9.1, 14
# y_true = logistic_4(x, a, b, c, d)
y_meas = np.array([0.4295, 0.6265, 0.9585, 1.2785, 1.6825, 1.8275, 2.102])

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
plt.xlabel('Concentration')
plt.ylabel('Density')
for i, (param, est) in enumerate(zip('ABCDE', p_optim[0])):
    plt.text(65, 1.75 - i * 0.1, '{} = {:.2f}'.format(param, est))
plt.text(x.min()+(x.max()-x.min()) * 0.05, y_meas.min()+(y_meas.max()-y_meas.min()), r"$Adj. R^2={:.3f}$".format(r_2))
plt.show()

# print(inv_logistic_4(np.array([1.07]), *p_optim[0]))
