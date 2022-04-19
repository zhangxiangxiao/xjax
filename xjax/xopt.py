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

import jax.tree_util as jtree
import jax.numpy as jnp


def optimizer(func):
    """Turn an optimizer into one that works on pytrees and add step counter."""
    def new_func(initial_params, *args, **kwargs):
        init, update = func(*args, **kwargs)
        flat_initial_params, treedef = jtree.tree_flatten(initial_params)
        flat_initial_states = [init(leaf) for leaf in flat_initial_params]
        # Add step count to states[0]
        initial_states = (0, flat_initial_states)
        def update(params, grads, states):
            flat_params = jtree.tree_leaves(params)
            flat_grads = jtree.tree_leaves(grads)
            step, flat_states = states
            flat_new_params, flat_new_states = zip(*(update(
                flat_params[i], flat_grads[i], flat_states[i], step)
                  for i in range(len(flat_states))))
            new_params = jtree.tree_unflatten(treedef, flat_new_params)
            new_states = (step + 1, flat_new_states)
            return new_params, new_states
        return update, initial_states
    return new_func


def callable_schedule(schedule):
    """Check and turn schedule into a callable object."""
    if callable(schedule):
        return schedule
    else:
        return lambda step:  rate


@optimizer
def SGD(rate=0.1, decay=0):
    """SGD optimizer."""
    rate = callable_schedule(rate)
    decay = callable_schedule(decay)
    def init(initial_params):
        return None
    def update(params, grads, states, step):
        grads = grads + decay(step) * params
        new_params = params - rate(step) * grads
        return new_params, states
    return init, update


@optimizer
def Momentum(rate=0.1, coeff=0.9, decay=0):
    """SGD with momentum optimizer."""
    rate = callable_schedule(rate)
    coeff = callable_schedule(coeff)
    decay = callable_schedule(decay)
    def init(initial_params):
        return jnp.zeros_like(initial_params)
    def update(params, grads, states, step):
        grads = grads + decay(step) * params
        new_states = coeff(step) * states + grads
        new_params = params - rate(step) * new_states
        return new_params, new_tates
    return init, update
