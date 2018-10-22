"""Utility functions for the mt-mkl."""
import collections

import numpy as np


def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] \
        if isinstance(lst, (list, np.ndarray)) else [lst]


def generate_index(X_list, y_list, cv):
    X_list_transpose = [X.transpose(1, 2, 0) for X in X_list]
    split = [cv.split(X, y) for X, y in zip(X_list_transpose, y_list)]
    n_splits = min(
        cv.get_n_splits(X, y, None) for X, y in zip(X_list_transpose, y_list))

    for _ in range(n_splits):
        yield zip(*[next(s) for s in split])


def update_rho(rho, rnorm, snorm, iteration=None, mu=10, tau_inc=2, tau_dec=2):
    """See Boyd pag 20-21 for details.

    Parameters
    ----------
    rho : float

    """
    if rnorm > mu * snorm:
        return tau_inc * rho
    elif snorm > mu * rnorm:
        return rho / tau_dec
    return rho


def namedtuple_with_defaults(typename, field_names, default_values=()):
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


convergence = namedtuple_with_defaults(
    'convergence', 'obj rnorm snorm e_pri e_dual precision')
