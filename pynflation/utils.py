from __future__ import print_function, division, absolute_import
import numpy as np


def nearest_idx(array, value):
    """
    For a given sorted array and a given value, find the index of the array element
    closest to the given value.
    :param np.ndarray array: one dimensional, sorted, numeric array
    :param value: single numeric value
    :return: int
        index
    """
    assert np.ndim(array) == 1, "input array expected to be one dimensional," \
                                "but np.ndim(array) = %d" % np.ndim(array)
    # assert is_sorted(array), "input array expected to be sorted."
    # assert type(value) is int or type(value) is float \
    #        or type(value) is type(np.array([0, 1])) or type(value) is type(np.array([0., 1.])), \
    #     "int or float expected for value input, " \
    #     "but value = %s, and type(value) = %s" % (value, type(value))
    return (np.abs(array - value)).argmin()
