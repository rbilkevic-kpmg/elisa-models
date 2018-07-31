

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
