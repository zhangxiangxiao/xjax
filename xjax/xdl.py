"""
Deep learning tools library for XJAX.
"""

from __future__ import absolute_import

from collections import namedtuple
import pickle

import jax
import jax.numpy as jnp
import numpy as np


TrainerTuple = namedtuple('TrainerTuple', ['train', 'states'])
TrainerStatesTuple = namedtuple(
    'TrainerStatesTuple', ['params', 'optimizer', 'model', 'metric'])

def Trainer(optimizer, model, metric=None):
    """Training tool for xjax models.

    Args:
      optimizer: an optimizer that was initialized from train_model's params.
      model: a model used for training.
      metric: an optional metric which is used to provide evaluation.


    Returns:
      train: a function that can be executed to train the model. The signature
        is `loss_outputs, eval_outputs, states = train(data, states, callback)`.
        `data` is a Python iterator that goes through sample inputs. `callback`
        is an optional function to be called at every optimization update,
        defined as `callback(step, states, inputs, net_outputs, loss_outputs,
        metric_outputs)`. `loss_outputs` and `metric_outputs` both are tuples of
        '(current_values, total_values)'.
      states: the initial states of trainer that contains model parameters
        and other states. `states[0]` is always the model parameters.
    """
    if metric is None:
        metric = (lambda inputs, net_outputs, states: None, None)
    update = optimizer[0]
    forward, backward = model[0], model[1]
    evaluate = metric[0]
    initial_states = TrainerStatesTuple(
        model[2], optimizer[1], model[3], metric[1])
    def train(data, states, callback=None):
        params, opt_states, model_states, metric_states = states
        total_loss_outputs, total_metric_outputs = None, None
        step = 0
        for inputs in data:
            grads, net_outputs, loss_outputs, model_states = backward(
                params, inputs, model_states)
            metric_outputs, metric_states = evaluate(
                inputs, net_outputs, metric_states)
            loss_outputs = jax.tree_map(lambda x: jnp.mean(x), loss_outputs)
            if total_loss_outputs is None:
                total_loss_outputs = loss_outputs
            else:
                total_loss_outputs = jax.tree_map(
                    lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                    total_loss_outputs, loss_outputs)
            metric_outputs = jax.tree_map(lambda x: jnp.mean(x), metric_outputs)
            if total_metric_outputs is None and metric_outputs is not None:
                total_metric_outputs = metric_outputs
            else:
                total_metric_outputs = jax.tree_map(
                    lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                    total_metric_outputs, metric_outputs)
            params, opt_states = update(params, grads, opt_states)
            if callback is not None:
                callback(opt_states[0] - 1, states, inputs, net_outputs,
                         (loss_outputs, total_loss_outputs),
                         (metric_outputs, total_metric_outputs))
            step = step + 1
        states = TrainerStatesTuple(
            params, opt_states, model_states, metric_states)
        return total_loss_outputs, total_metric_outputs, states
    return TrainerTuple(train, initial_states)


TesterTuple = namedtuple('TesterTuple', ['test', 'states'])
TesterStatesTuple = namedtuple(
    'TesterStatesTuple', ['params', 'model', 'metric'])

def Tester(model, metric=None):
    """Testing tool for xjax models.

    Args:
      model: a model used for testing.
      metric: an optional metric which is used to provide evaluation.


    Returns:
      test: a function that can be executed to test the model. The signature
        is `loss_outputs, metric_outputs, states = test(data, states, callback)`.
        `data` is a Python iterator that goes through sample inputs. `callback`
        is an optional function to be called at every optimization update,
        defined as `callback(step, states, inputs, net_outputs, loss_outputs,
        metric_outputs)`. `loss_outputs` and `metric_outputs` both are tuples of
        '(current_values, total_values)'.
        defined as `callback(step, states, inputs, outputs, losses, totals)`. 
      states: the initial states of tester that contains model parameters
        and other states. `states[0]` is always the model parameter.
    """
    if metric is None:
        metric = (lambda inputs, net_outputs, states: (None, None), None)
    forward = model[0]
    evaluate = metric[0]
    initial_states = TesterStatesTuple(model[2], model[3], metric[1])
    def test(data, states, callback=None):
        params, model_states, metric_states = states
        total_loss_outputs, total_metric_outputs = None, None
        step = 0
        for inputs in data:
            net_outputs, loss_outputs, model_states = forward(
                params, inputs, model_states)
            metric_outputs, metric_states = evaluate(
                inputs, net_outputs, metric_states)
            loss_outputs = jax.tree_map(lambda x: jnp.mean(x), loss_outputs)
            if total_loss_outputs is None:
                total_loss_outputs = loss_outputs
            else:
                total_loss_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*y,
                    total_loss_outputs, loss_outputs)
            metric_outputs = jax.tree_map(lambda x: jnp.mean(x), metric_outputs)
            if total_metric_outputs is None and metric_outputs is not None:
                total_metric_outputs = metric_outputs
            else:
                total_metric_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*y,
                    total_metric_outputs, metric_outputs)
            if callback is not None:
                callback(step, inputs, states, net_outputs,
                         (loss_outputs, total_loss_outputs),
                         (metric_outputs, total_metric_outputs))
            step = step + 1
        states = TesterStatesTuple(params, model_states, metric_states)
        return total_loss_outputs, total_metric_outputs, states
    return TesterTuple(test, initial_states)


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
    """Loads states from bytes."""
    states = pickle.loads(data)
    return jax.tree_map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, states)
