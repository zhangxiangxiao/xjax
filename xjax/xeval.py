"""
Evaluation library for JAX.

This module implements evaluators for xjax.
"""

from __future__ import absolute_import

from collections import namedtuple

import jax
import jax.numpy as jnp
from xjax import xnn
from xjax.xnn import vectorize_states, postprocess_states

EvaluatorTuple = namedtuple('Evaluator', ['evaluate', 'states'])


def Evaluator(module):
    """Generic evaluator which is simply an xjax.xnn module.
    Args:
      module: an xjax.xnn module whose ouputs will be used as evluation outputs.

    Returns:
      evaluate: the evaluate function.
      states: the initial states of evaluator, same as module states in xnn.
    """
    forward, params, initial_states = module
    def evaluate(inputs, net_outputs, states):
        return forward(params, [inputs, net_outputs], states)
    return EvaluatorTuple(evaluate, initial_states)


def ClassEvaluator():
    """Classification evaluator. Assuming _, labels = inputs."""
    def evaluate(inputs, net_outputs, states):
        net_labels = jnp.argmax(net_outputs, axis=-1)
        return jnp.mean(jnp.equal(inputs[1], net_labels)), states
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
    evaluate_v = map_func(evaluate, *args, **kwargs)
    def evaluate(inputs, net_outputs, states):
        outputs, states = evaluate_v(inputs, net_outputs, states)
        new_states = postprocess_states(states, size)
        return outputs, new_states
    return EvaluatorTuple(evaluate, initial_states)

vmap = partial(vectorize, jax.vmap)
pmap = partial(vectorize, jax.pmap)
