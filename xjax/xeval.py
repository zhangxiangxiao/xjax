"""
Evaluation library for JAX.

This module implements evaluators for xjax.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtree
from xjax import xnn


EvaluatorTuple = namedtuple('Evaluator', ['evaluate', 'states'])


def Evaluator(module):
    """Generic evaluator which is simply an xjax.xnn module. Assuming
      net_inputs, net_targets = inputs.

    Args:
      module: an xjax.xnn module whose ouputs will be used as evluation outputs.

    Returns:
      evaluate: the evaluate function.
      states: the initial states of evaluator, same as module states in xnn.
    """
    forward, params, initial_states = module
    def evaluate(inputs, net_outputs, states):
        return forward(params, [inputs[1], net_outputs], states)
    return EvaluatorTuple(evaluate, initial_states)


def BinaryEval():
    """Binary classification evaluator. Assuming _, labels = inputs. labels is
       either -1 or 1."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.where(net_outputs > 0, 1, -1)
        return jnp.mean(jnp.equal(inputs[1], net_labels)), states
    return EvaluatorTuple(evaluate, None)


def ClassEval():
    """Classification evaluator. Assuming _, labels = inputs."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.argmax(net_outputs, axis=-1)
        return jnp.mean(jnp.equal(inputs[1], net_labels)), states
    return EvaluatorTuple(evaluate, None)


def CategoricalEval():
    """Categorical classification evaluator. Assuming _, targets = inputs."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.argmax(net_outputs, axis=-1)
        tar_labels = jnp.argmax(inputs[1], axis=-1)
        return jnp.mean(jnp.equal(net_labels, tar_labels)), states
    return EvaluatorTuple(evaluate, None)


def vectorize(evaluator, map_func=jax.vmap, *args, **kwargs):
    """Vectorize the evaluator.

    Args:
      evaluator: the evaluator to be vectorized.
      map_func: jax.vmap or jax.pmap

    Returns:
      evaluate: the vectorized evaluate function.
      states: vectorized states.
    """
    eval_evaluate, initial_states = evaluator
    # Map over inputs and states
    evaluate_v = map_func(eval_evaluate, *args, **kwargs)
    def evaluate(inputs, net_outputs, states):
        batch = jtree.tree_leaves(inputs)[0].shape[0]
        states = xnn.vectorize_states(states, batch)
        outputs, states = evaluate_v(inputs, net_outputs, states)
        states = xnn.unvectorize_states(states)
        return outputs, states
    return EvaluatorTuple(evaluate, initial_states)


def jit(evaluator, *args, **kwargs):
    """Set up the evaluator for JIT.

    Args:
      evaluator: an xeval evaluator.

    Returns:
      jit_evaluator: JIT'ed evaluator.
    """
    evaluate, states = evaluator
    return EvaluatorTuple(jax.jit(evaluate, *args, **kwargs), states)
