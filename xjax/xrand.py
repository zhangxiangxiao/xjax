"""Random generators for XJAX.

xjax.xrand maintains an internal rng that is initialized by
jax.random.PRNGKey(time.time_ns()). It has the following functions:
xrand.split(num): split the internal rng and return new ones.
xrand.get(): get the internal rng.
xrand.set(rng): set the internal rng.
"""
import time

import jax.random as jrand


_rng = jrand.PRNGKey(time.time_ns())


def split(num=None):
    global _rng
    rng, _rng = jrand.split(_rng)
    if num is not None and num > 1:
        rng = jrand.split(rng, num)
    return rng


def get():
    return _rng


def set(rng):
    global _rng
    _rng = rng
