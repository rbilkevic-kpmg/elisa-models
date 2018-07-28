import numpy as np
from scipy.optimize import least_squares


def logistic_4(x, a, b, c, d):
    """
    4PL logistic equation
    :param float x: signal value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :return: float
    """
    return (a - d) / (1 + (x / c) ** b) + d


def residuals(p, y, x):
    """
    Deviations of the data from
    :param tuple p: tuple of 4PL parameters
    :param y: response value
    :param x: signal value
    :return: float
    """
    a, b, c, d = p
    err = y - logistic_4(x, a, b, c, d)
    return err


def inv_logistic_4(y, a, b, c, d):
    """
    Inverse 4PL logistic equation
    :param float y: response value
    :param float a: minimum asymptote
    :param float b: Hills' slope
    :param float c: inflection slope
    :param float d: maximum asymptote
    :return: float
    """
    return c * ((a - y) / (y - d)) ** (1 / b)


# Initial set of parameters
init_p = [0.5, 0.25, 0.4, 1]
