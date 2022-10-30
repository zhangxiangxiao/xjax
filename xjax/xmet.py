"""
Metric library for XJAX.

This module implements metrics for xjax.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
from xjax import xnn


MetricTuple = namedtuple('MetricTuple', ['evaluate', 'states'])

def Metric(module):
    """Generic metric which is simply an xjax.xnn module.

    Args:
      module: an xjax.xnn module whose ouputs will be used as evluation outputs.

    Returns:
      evaluate: the evaluate function.
      states: the initial states of metric, same as module states in xnn.
    """
    forward, params, initial_states = module
    def evaluate(inputs, net_outputs, states):
        return forward(params, (inputs, net_outputs), states)
    return MetricTuple(evaluate, initial_states)

def Binary():
    """Binary classification metric. Assuming labels = inputs[-1]."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.where(net_outputs > 0, 1, -1)
        return jnp.mean(jnp.equal(inputs[-1], net_labels)), states
    return MetricTuple(evaluate, None)

def MultiClass():
    """Multi-class metric. Assuming labels = inputs[-1]."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.argmax(net_outputs, axis=-1)
        return jnp.mean(jnp.equal(inputs[-1], net_labels)), states
    return MetricTuple(evaluate, None)

def Categorical():
    """Categorical classification metric. Assuming targets = inputs[-1]."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.argmax(net_outputs, axis=-1)
        tar_labels = jnp.argmax(inputs[-1], axis=-1)
        return jnp.mean(jnp.equal(net_labels, tar_labels)), states
    return MetricTuple(evaluate, None)


def vectorize(metric, map_func=jax.vmap, *args, **kwargs):
    """Vectorize the metric.

    Args:
      metric: the metric to be vectorized.
      map_func: jax.vmap or jax.pmap

    Returns:
      evaluate: the vectorized evaluate function.
      states: vectorized states.
    """
    eval_evaluate, initial_states = metric
    # Map over inputs and states
    evaluate_v = map_func(eval_evaluate, *args, **kwargs)
    def evaluate(inputs, net_outputs, states):
        batch = jtree.tree_leaves(inputs)[0].shape[0]
        states = xnn.vectorize_states(states, batch)
        outputs, states = evaluate_v(inputs, net_outputs, states)
        states = xnn.unvectorize_states(states)
        return outputs, states
    return MetricTuple(evaluate, initial_states)


def jit(metric, *args, **kwargs):
    """Set up the metric for JIT.

    Args:
      metric: an xeval metric.

    Returns:
      jit_metric: JIT'ed metric.
    """
    evaluate, states = metric
    return MetricTuple(jax.jit(evaluate, *args, **kwargs), states)
