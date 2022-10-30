"""
Deep learning tools for xjax.
"""

from __future__ import absolute_import

from collections import namedtuple
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def train(data, model, optimizer, metric=None, callback=None):
    """Training function for xjax models.

    Args:
      data: a Python iterable object that goes through samples inputs.
      model: an xjax model.
      optimizer: an xjax optimizer.
      metric: an optional xjax metric.
      callback: an optional function to be called after every optimization step.
        It should be defined as `callback(step, states, values, currents,
        totals)`. If `metric` is `None`, `states = (model_states,
        optimizer_states)`, and `currents = loss_outputs`. If `metric` is not
        `None`, `states = (model_states, optimizer_states, metric_states)` and
        `currents = (loss_outputs, metric_outputs)`. `values` is `(inputs,
        grads, net_outputs)`, `totals` is a running average computed from
        `currents`.

    Returns:
      model: new model with updated parameters and states after training.
      optimizer: new optimizer with updated states after training.
      metric: only returned if `metric` is not `None`. New metric with updates
        states after training.
      totals: averaged outputs from model loss and metric. Computed from
        `loss_outputs` if `metric` is `None`, and `(loss_outputs,
        metric_outputs)` otherwise.
    """
    forward, backward, params, model_states = model
    update, optimizer_states = optimizer
    if metric:
        evaluate, metric_states = metric
    totals = None
    step = 0
    for inputs in data:
        grads, net_outputs, loss_outputs, model_states = backward(
            params, inputs, model_states)
        params, optimizer_states = update(params, grads, optimizer_states)
        if metric:
            metric_outputs, metric_states = evaluate(
                inputs, net_outputs, metric_states)
            currents = (loss_outputs, metric_outputs)
        else:
            currents = loss_outputs
        totals = currents if not totals else jax.tree_map(
            lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
            totals, currents)
        if callback:
            values = (inputs, grads, net_outputs)
            if metric:
                states = (model_states, optimizer_states, metric_states)
            else:
                states = (model_states, optimizer_states)
            callback(step, states, values, currents, totals)
        step = step + 1
    model = type(model)(forward, backward, params, model_states)
    optimizer = type(optimizer)(update, optimizer_states)
    if metric:
        metric = type(metric)(evaluate, metric_states)
        return model, optimizer, metric, totals
    else:
        return model, optimizer, totals

def test(data, model, metric=None, callback=None):
    """Testing function for xjax models.

    Args:
      data: a Python iterable object that goes through samples inputs.
      model: an xjax model.
      metric: an optional xjax metric.
      callback: an optional function to be called after every testing step.
        It should be defined as `callback(step, states, values, currents,
        totals)`. If `metric` is `None`, `states = model_states`, and `currents
        = loss_outputs`. If `metric` is not `None`, `states = (step,
        model_states, metric_states)`, and `currents = (loss_outputs,
        metric_outputs)`. `values` is `(inputs, net_outputs)`, `totals` is a
        running average computed from `currents`.

    Returns:
      model: new model with updated states after testing.
      metric: only returned if `metric` is not `None`. New metric with updates
        states after testing.
      totals: averaged outputs from model loss and metric. Computed from
        `loss_outputs` if `metric` is `None`, and `(loss_outputs,
        metric_outputs)` otherwise.
    """
    forward, backward, params, model_states = model
    if metric:
        evaluate, metric_states = metric
    totals = None
    step = 0
    for inputs in data:
        net_outputs, loss_outputs, model_states = forward(
            params, inputs, model_states)
        if metric:
            metric_outputs, metric_states = evaluate(
                inputs, net_outputs, metric_states)
            currents = (loss_outputs, metric_outputs)
        else:
            currents = loss_outputs
        totals = currents if not totals else jax.tree_map(
            lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
            totals, currents)
        if callback:
            values = (inputs, net_outputs)
            if metric:
                states = (model_states, metric_states)
            else:
                states = model_states
            callback(step, states, values, currents, totals)
        step = step + 1
    model = type(model)(forward, backward, params, model_states)
    if metric:
        metric = type(metric)(evaluate, metric_states)
        return model, metric, totals
    else:
        return model, totals


def dump(states, fd):
    """Serialize states to file."""
    states = jax.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, states)
    return pickle.dump(states, fd)

def dumps(states):
    """Serialize states to bytes."""
    states = jax.tree_map(
        lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, states)
    return pickle.dumps(states)

def load(fd):
    """Load states from file."""
    states = pickle.load(fd)
    return jax.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, states)

def loads(data):
    """Load states from bytes."""
    states = pickle.loads(data)
    return jax.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, states)
