"""
Functinal optimization library for JAX.

Design principle: an xjax optimizer is a function that returns 2 objects:
  update: the update function.
  initial_states: initial states that update will modify; should be picklable.

The signature of the update function should be
new_params, new_states = update(params, grads, states)
states[0] is always the current step count.

We provide an @optimizer decorator that turns an optimizer into one that works
on pytrees and also adds the step counter. It should be applied to a function
that retuns 2 objects:
  init: the state initialize function
  update: the update function

The signature of the init function should be
initial_states = init(initial_params)

The signature of the update function should be
new_params, states = update(params, grads, states, step)
Note the additional `step` argument compared to the decorated optimizer.
"""

from __future__ import absolute_import

from functools import wraps, partial
from collections import namedtuple

import jax
import jax.tree_util as jtree
import jax.numpy as jnp


OptimizerTuple = namedtuple('OptimizerTuple', ['update', 'states'])
OptimizerStatesTuple = namedtuple(
    'OptimizerStatesTuple', ['step', '_1'], rename=True)


def tree_init(init):
    """Make init function work with pytrees and add step counter."""
    @wraps(init)
    def wrapped_init(initial_params):
        return (0, jax.tree_map(init, initial_params))
    return wrapped_init

def tree_update(update):
    """Make an update function work with pytrees and use step counter"""
    @wraps(update)
    def wrapped_update(params, grads, states):
        step = states[0]
        flat_params, treedef = jtree.tree_flatten(params)
        flat_grads = treedef.flatten_up_to(grads)
        flat_states = treedef.flatten_up_to(states[1])
        flat_new_params, flat_new_states = zip(*(update(
            step, *leaf) for leaf in zip(flat_params, flat_grads, flat_states)))
        new_params = treedef.unflatten(flat_new_params)
        new_states = (step + 1, treedef.unflatten(flat_new_states))
        return new_params, new_states
    return wrapped_update


def callable_schedule(schedule):
    """Check and turn schedule into a callable object."""
    if callable(schedule):
        return schedule
    else:
        return lambda step: schedule


def SGD(initial_params, rate=0.1, decay=0):
    """SGD optimizer."""
    rate = callable_schedule(rate)
    decay = callable_schedule(decay)
    @tree_update
    def update(step, params, grads, states):
        grads = grads + decay(step) * params
        new_params = params - rate(step) * grads
        return new_params, states
    @tree_init
    def init(initial_params):
        return None
    initial_states = init(initial_params)
    return update, initial_states


def Momentum(initial_params, rate=0.1, coeff=0.9, decay=0):
    """SGD with momentum optimizer."""
    rate = callable_schedule(rate)
    coeff = callable_schedule(coeff)
    decay = callable_schedule(decay)
    @tree_update
    def update(step, params, grads, states):
        grads = grads + decay(step) * params
        new_states = coeff(step) * states + grads
        new_params = params - rate(step) * new_states
        return new_params, new_states
    @tree_init
    def init(initial_params):
        return jnp.zeros_like(initial_params)
    initial_states = init(initial_params)
    return update, initial_states
