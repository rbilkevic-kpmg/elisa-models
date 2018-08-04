import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import os

#plt.style.use('seaborn-paper')


from utils import (
    residuals_func,
    r_squared_adj
)
from data import ElisaData


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


# File
file_name = [f for f in os.listdir('input\\') if f.endswith('.xlsx')][0]

# Data
elisa = ElisaData('input\\' + file_name)
x = np.array([60, 30, 15, 7.5, 3.75, 1.875, 0.9375])
x_range = x.max() - x.min()
x_graph = np.linspace(x.min() - x_range * 0.25, x.max() + x_range * 0.25, 100)
# a, b, c, d, e = 0.5, 2.5, 8, 9.1, 14
# y_true = logistic_4(x, a, b, c, d)
y_meas = np.array(elisa.STANDARDS_DATA)
y_range = y_meas.max() - y_meas.min()

# Initial set of parameters
p_init = np.array([2, 4, 9, 5])

# Fit equation using least squares
p_optim = leastsq(residuals_func(logistic_4), p_init, args=(y_meas, x))
print(p_optim[0])
y_pred = logistic_4(x, *p_optim[0])
r_2 = r_squared_adj(y_meas, y_pred, len(x), len(p_optim))

# Plot results
plt.plot(x_graph, logistic_4(x_graph, *p_optim[0]), x, y_meas, 'o')
plt.legend(['Model Fit', 'Measured'])
plt.title('4 Point Linear Regression')
plt.xlabel('8-OHdG Concentration, ng/ml')
plt.ylabel('Net Optical Density')

plt.text(x.min() + (x.max() - x.min()) * 0.05, y_meas.min() + (y_meas.max() - y_meas.min()),
         r"$Adj. R^2={:.3f}$".format(r_2))

plt.show()

concentrations = inv_logistic_4(np.array(elisa.DATA), *p_optim[0])
elisa.write_concentrations(concentrations)
