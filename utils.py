import numpy as np


def residuals_func(func):
    """
    Deviations of the data from

    :param func: function
    :return: func
    """
    def residuals(p, y, x):
        """

        :param tuple p: tuple of 4PL parameters
        :param nd_array y: response value
        :param nd_array x: signal value
        :return: nd_array
        """
        err = y - func(x, *p)
        return err
    return residuals


def r_squared_adj(y, y_pred, n, k):
    """

    :param y:
    :param y_pred:
    :return:
    """
    y_mean = np.mean(y)
    ss_tot = np.dot((y - y_mean), (y - y_mean).transpose())
    ss_reg = np.dot((y_pred - y_mean), (y_pred - y_mean).transpose())

    return 1 - (1 - ss_reg / ss_tot) * (n - 1) / (n - k - 1)