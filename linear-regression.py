import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import os

from utils import (
    residuals_func,
    r_squared_adj
)
from data import CRPData
plt.style.use('seaborn-paper')


def linear(x, a, b):
    """
    4PL logistic equation

    :param nd_array x: signal value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :return: nd_array
    """
    return a + b * x


def inv_linear(y, a, b):
    """
    Inverse 4PL logistic equation

    :param nd_array y: response value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :return: nd_array
    """
    return (y - a) / b


# File
file_name = [f for f in os.listdir('input\\') if f.endswith('.xlsx')][0]

# Data
crp = CRPData('input\\' + file_name)
x = np.array([0, 3000, 1500, 750, 375, 187.5, 93.75])
x_range = x.max() - x.min()
x_graph = np.linspace(x.min() - x_range * 0.05, x.max() + x_range * 0.05, 100)
# a, b, c, d, e = 0.5, 2.5, 8, 9.1, 14
# y_true = logistic_4(x, a, b, c, d)
y_meas = np.array(crp.STANDARDS_DATA)
y_range = y_meas.max() - y_meas.min()

# Initial set of parameters
p_init = np.array([1, 5])

# Fit equation using least squares
p_optim = leastsq(residuals_func(linear), p_init, args=(y_meas, x))
print(p_optim[0])
y_pred = linear(x, *p_optim[0])
r_2 = r_squared_adj(y_meas, y_pred, len(x), len(p_optim))

# Plot results
plt.plot(x_graph, linear(x_graph, *p_optim[0]), x, y_meas, 'o')
plt.legend(['Model Fit', 'Measured'])
plt.title('Linear Regression')
plt.xlabel('CRP Concentration, pg/ml')
plt.ylabel('Net Optical Density')

plt.text(x.min() + (x.max() - x.min()) * 0.15, y_meas.min() + (y_meas.max() - y_meas.min()),
         r"$Adj. R^2={:.3f}$".format(r_2))
plt.show()

concentrations = inv_linear(np.array(crp.DATA), *p_optim[0])
crp.write_concentrations(concentrations)
