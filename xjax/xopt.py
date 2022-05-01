"""
Functinal optimization library for JAX.

Design principle: an xjax optimizer is a function that returns 2 objects:
  update: the update function.
  initial_states: initial states that update will modify; should be picklable.

The signature of the update function should be
new_params, new_states = update(params, grads, states)
states[0] is always the current step count.
"""

from __future__ import absolute_import
from collections import namedtuple
from functools import partial, wraps
import math

import jax
import jax.numpy as jnp
import jax.tree_util as jtree


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
    """SGD optimizer. Supports both dense and segment gradients."""
    rate = callable_schedule(rate)
    decay = callable_schedule(decay)
    @tree_update
    def update(step, params, grads, states):
        if isinstance(grads, jnp.ndarray):
            # Dense gradients.
            grads = grads + decay(step) * params
            new_params = params - rate(step) * grads
        else:
            # Segment gradients.
            index, grads_value = grads
            params_value = jnp.take(params, index, axis=0)
            grads_value = grads_value + decay(step) * params_value
            # Caveat: weight decay updates not in sequence on repeated indices.
            new_params = params.at[index].add(-rate(step) * grads_value)
        return new_params, states
    @tree_init
    def init(initial_params):
        return None
    initial_states = init(initial_params)
    return OptimizerTuple(update, initial_states)


def Momentum(initial_params, rate=0.1, coeff=0.9, decay=0):
    """SGD with momentum optimizer."""
    rate = callable_schedule(rate)
    coeff = callable_schedule(coeff)
    decay = callable_schedule(decay)
    @tree_update
    def update(step, params, grads, states):
        if isinstance(grads, jnp.ndarray):
            # Dense gradients.
            grads = grads + decay(step) * params
            new_states = coeff(step) * states + grads
            new_params = params - rate(step) * new_states
        else:
            # Segment gradients.
            index, grads_value = grads
            params_value = jnp.take(params, index, axis=0)
            states_value = jnp.take(states, index, axis=0)
            grads_value = grads_value + decay(step) * params_value
            new_states_value = coeff(step) * states_value + grads_value
            # Caveat: overwriten velocity on repeated indices.
            new_states = states.at[index].set(new_states_value)
            # Caveat: weight decay updates not in sequence on repeated indices.
            new_params = params.at[index].add(-rate(step) * new_states_value)
        return new_params, new_states
    @tree_init
    def init(initial_params):
        return jnp.zeros_like(initial_params)
    initial_states = init(initial_params)
    return OptimizerTuple(update, initial_states)


def vectorize(optimizer):
    """Vectorize the optimizer for vmap or pmap gradients.
    Gradients are averaged over the mapped dims.
    """
    opt_update, initial_states = optimizer
    def update(params, grads, states):
        def leaf_reduce(_params, _grads):
            if isinstance(_grads, jnp.ndarray):
                # Dense gradient.
                _grads = jnp.reshape(_grads, (-1,) + _params.shape)
                return jnp.mean(_grads, axis=0)
            else:
                # Segment gradient.
                _index, _grads_value = _grads
                _grads_value = _grads_value / math.prod(_index.shape[0:-1])
                _grads_value = jnp.reshape(
                    _grads_value, (-1,) + _params.shape[1:])
                _index = jnp.reshape(_index, (-1,))
                return (_index, _grads_value)
        grads = jax.tree_map(leaf_reduce, params, grads)
        return opt_update(params, grads, states)
    return OptimizerTuple(update, initial_states)

vmap = vectorize
# pmap = vectorize
