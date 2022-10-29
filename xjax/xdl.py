"""
Deep learning tools library for XJAX.
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
        It should be defined as `callback(step, states, inputs, net_outputs,
        loss_outputs, metric_outputs)`. Both `loss_outputs` and `metric_outputs`
        are tuples of `(current_values, total_values)`.

    Returns:
      loss_outputs: running-averaged loss outputs.
      metric_outputs: running-averaged metric outputs.
      model: new model with updated parameters and states after training.
      optimizer: new optimizer with updated states after training.
      metric: new metric with updated states after training.
    """
    forward, backward, params, model_states = model
    update, optimizer_states = optimizer
    evaluate, metric_states = metric if metric else (
        lambda inputs, net_outputs, states: (None, None), None)
    total_loss_outputs, total_metric_outputs = None, None
    step = 0
    for inputs in data:
        grads, net_outputs, loss_outputs, model_states = backward(
            params, inputs, model_states)
        metric_outputs, metric_states = evaluate(
            inputs, net_outputs, metric_states)
        loss_outputs = jax.tree_map(lambda x: jnp.mean(x), loss_outputs)
        total_loss_outputs = loss_outputs if not total_loss_outputs else (
            jax.tree_map(
                lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                total_loss_outputs, loss_outputs))
        metric_outputs = jax.tree_map(lambda x: jnp.mean(x), metric_outputs)
        total_metric_outputs = metric_outputs if not total_metric_outputs else (
            jax.tree_map(
                lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                total_metric_outputs, metric_outputs))
        params, optimizer_states = update(params, grads, optimizer_states)
        if callback:
            callback(optimizer_states[0] - 1, inputs, net_outputs,
                     (loss_outputs, total_loss_outputs),
                     (metric_outputs, total_metric_outputs))
        step = step + 1
    model = type(model)(forward, backward, params, model_states)
    optimizer = type(optimizer)(update, optimizer_states)
    if metric:
        metric = type(metric)(evaluate, metric_states)
    return total_loss_outputs, total_metric_outputs, model, optimizer, metric 

def test(data, model, metric, callback=None):
    """Testing function for xjax models.

    Args:
      data: a Python iterable object that goes through samples inputs.
      model: an xjax model.
      optimizer: an xjax optimizer.
      metric: an optional xjax metric.
      callback: an optional function to be called after every optimization step.
        It should be defined as `callback(step, states, inputs, net_outputs,
        loss_outputs, metric_outputs)`. Both `loss_outputs` and `metric_outputs`
        are tuples of `(current_values, total_values)`.

    Returns:
      loss_outputs: running-averaged loss outputs.
      metric_outputs: running-averaged metric outputs.
      model: new model with updated states after testing.
      metric: new metric with updated states after testing.
    """
    forward, backward, params, model_states = model
    evaluate, metric_states = metric if metric else (
        lambda inputs, net_outputs, states: (None, None), None)
    total_loss_outputs, total_metric_outputs = None, None
    step = 0
    for inputs in data:
        net_outputs, loss_outputs, model_states = forward(
            params, inputs, model_states)
        metric_outputs, metric_states = evaluate(
            inputs, net_outputs, metric_states)
        loss_outputs = jax.tree_map(lambda x: jnp.mean(x), loss_outputs)
        total_loss_outputs = loss_outputs if not total_loss_outputs else (
            jax.tree_map(
                lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                total_loss_outputs, loss_outputs))
        metric_outputs = jax.tree_map(lambda x: jnp.mean(x), metric_outputs)
        total_metric_outputs = metric_outputs if not total_metric_outputs else (
            jax.tree_map(
                lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                total_metric_outputs, metric_outputs))
        if callback:
            callback(step, inputs, net_outputs,
                     (loss_outputs, total_loss_outputs),
                     (metric_outputs, total_metric_outputs))
        step = step + 1
    model = type(model)(forward, backward, params, model_states)
    if metric:
        metric = type(metric)(evaluate, metric_states)
    return total_loss_outputs, total_metric_outputs, model, metric


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
