"""
Deep learning tools library for JAX.

The tools in this module is based on Learner, which contains the code for
training, testing, evaluation, and serialization of a deep learning model.
"""

from __future__ import absolute_import

from collections import namedtuple
import pickle

import jax
import jax.numpy as jnp
import numpy as np


LearnerTuple = namedtuple('LearnerTuple', ['train', 'test', 'states'])
LearnerStatesTuple = namedtuple(
    'LearnerStatesTuple', ['params', '_1', '_2', '_3', '_4'], rename=True)


def Learner(optimizer, train_model, test_model=None, evaluator=None):
    """Training and testing tool for xjax models.

    Args:
      optimizer: an optimizer that was initialized from train_model's params.
      train_model: a model used for training.
      test_model: a model used for testing. It will use train_model's params.
        This is useful if the `forward` function of the training model is
        different at testing time, such as when using dropout. `train_model` is
        used if set to `None`.
      evaluator: an (evaluate, states) tuple which is used to provide evaluation
        in addition to the loss_outputs from models. The `evaluate` function is
        defined as `outputs, states = evaluate(inputs, net_outputs, states)`.


    Returns:
      train: a function that can be executed to train the model. The signature
        is `loss_outputs, eval_outputs, states = train(data, states, callback=None)`.
        `data` is a Python iterator that goes through sample inputs. `callback`
        is a function to be called at every optimization update, defined as
        `callback(step, params, grads, inputs, net_outputs, loss_outputs,
        eval_outputs, total_loss_outputs, total_eval_outputs)`.
      test: a function that can be called to test the model. The signature is
        `loss_outputs, eval_outputs, states = test(data, states, callback=None)`.
        `data` is a Python iterator that goes through sample inputs. `callback`
        is a function to be called at every optimization update, defined as
        `callback(step, inputs, net_outputs, loss_outputs, eval_outputs,
        total_loss_outputs, total_eval_outputs)`.
      states: the initial states of learner that contains model parameters
        and other states. `states[0]` is always the model parameter.
    """
    if test_model is None:
        test_model = train_model
    if evaluator is None:
        evaluator = (lambda inputs, net_outputs, states: (None, None), None)
    update = optimizer[0]
    backward = train_model[1]
    forward = test_model[0]
    evaluate = evaluator[0]
    initial_states = LearnerStatesTuple(
        train_model[2], optimizer[1], train_model[3], test_model[3],
        evaluator[1])
    def train(data, states, callback=None):
        params = states[0]
        opt_states, model_states, eval_states = states[1], states[2], states[4]
        total_loss_outputs, total_eval_outputs = None, None
        step = 0
        for inputs in data:
            grads, net_outputs, loss_outputs, model_states = backward(
                params, inputs, model_states)
            eval_outputs, eval_states = evaluate(
                inputs, net_outputs, eval_states)
            if total_loss_outputs is None:
                total_loss_outputs = jax.tree_map(
                    lambda x: jnp.mean(x), loss_outputs)
            else:
                total_loss_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*jnp.mean(y),
                    total_loss_outputs, loss_outputs)
            if total_eval_outputs is None and eval_outputs is not None:
                total_eval_outputs = jax.tree_map(
                    lambda x: jnp.mean(x), eval_outputs)
            else:
                total_eval_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*jnp.mean(y),
                    total_eval_outputs, eval_outputs)
            params, opt_states = update(params, grads, opt_states)
            if callback is not None:
                callback(opt_states[0] - 1, params, grads, inputs, net_outputs,
                         loss_outputs, eval_outputs, total_loss_outputs,
                         total_eval_outputs)
            step = step + 1
        states = LearnerStatesTuple(
            params, opt_states, model_states, states[3], eval_states)
        return total_loss_outputs, total_eval_outputs, states
    def test(data, states, callback=None):
        params = states[0]
        opt_states, model_states, eval_states = states[1], states[3], states[4]
        total_loss_outputs, total_eval_outputs = None, None
        step = 0
        for inputs in data:
            net_outputs, loss_outputs, model_states = forward(
                params, inputs, model_states)
            eval_outputs, eval_states = evaluate(
                inputs, net_outputs, eval_states)
            if total_loss_outputs is None:
                total_loss_outputs = jax.tree_map(
                    lambda x: jnp.mean(x), loss_outputs)
            else:
                total_loss_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*jnp.mean(y),
                    total_loss_outputs, loss_outputs)
            if total_eval_outputs is None and eval_outputs is not None:
                total_eval_outputs = jax.tree_map(
                    lambda x: jnp.mean(x), eval_outputs)
            else:
                total_eval_outputs = jax.tree_map(
                    lambda x, y: step/(step+1)*x + 1/(step+1)*jnp.mean(y),
                    total_eval_outputs, eval_outputs)
            if callback is not None:
                callback(step, inputs, net_outputs, loss_outputs, eval_outputs,
                         total_loss_outputs, total_eval_outputs)
            step = step + 1
        states = LearnerStatesTuple(
            params, opt_states, states[2], model_states, eval_states)
        return total_loss_outputs, total_eval_outputs, states
    return LearnerTuple(train, test, initial_states)


def dump(states, fd):
    """Serialize states to file."""
    return pickle.dump(states, fd)


def dumps(states):
    """Serialize states to bytes."""
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
