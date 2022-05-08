"""
Evaluation library for JAX.

This module implements evaluators for xjax.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
from xjax import xnn
from xjax.xnn import vectorize_states, postprocess_states

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


def vectorize(map_func, evaluator, size, *args, **kwargs):
    """Vectorize the evaluator.

    Args:
      map_func: jax.vmap or jax.pmap
      evaluator: the evaluator to be vectorized.
      size: the batch size.

    Returns:
      evaluate: the vectorized evaluate function.
      states: vectorized states.
    """
    eval_evaluate, eval_states = evaluator
    initial_states = vectorize_states(eval_states, size)
    # Map over inputs and states
    evaluate_v = map_func(eval_evaluate, *args, **kwargs)
    def evaluate(inputs, net_outputs, states):
        outputs, states = evaluate_v(inputs, net_outputs, states)
        new_states = postprocess_states(states, size)
        return outputs, new_states
    return EvaluatorTuple(evaluate, initial_states)

def vmap(evaluator, size, *args, **kwargs):
    return vectorize(jax.vmap, evaluator, size, *args, **kwargs)

def pmap(evaluator, size, *args, **kwargs):
    return vectorize(jax.pmap, evaluator, size, *args, **kwargs)


def jit(evaluator, *args, **kwargs):
    """Set up the evaluator for JIT.

    Args:
      evaluator: an xeval evaluator.

    Returns:
      jit_evaluator: JIT'ed evaluator.
    """
    evaluate, states = evaluator
    return EvaluatorTuple(jax.jit(evaluate, *args, **kwargs), states)
