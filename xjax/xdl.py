"""
Deep learning tools library for JAX.

The tools in this module include trainer, tester, serializer, and a Learner
function to put everything together.
"""

from __future__ import absolute_import

from collections import namedtuple

import jax
from xjax import xmod


LearnerTuple = namedtuple('LearnerTuple', ['train', 'test', 'states'])


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
        is `outputs, states = train(data, states, callback=None)`. `data` is a
        Python iterator that goes through sample inputs. `callback` is a
        function to be called at every optimization update, defined as
        `callback(model, backward_outputs, optimizer, evaluator, eval_outputs)`.
        `outputs` is a tuple `(loss_outputs, evaluate_outputs)` which are
        averaged over all samples in `data`.
      test: a function that can be called to test the model. The signature is
        `outputs, states = test(data, states, callback=None)`. `data` is
        a Python iterator that goes through sample inputs. `callback` is a
        function to be called at every optimization update, defined as
        `callback(model, forward_outputs, optimizer, evaluator, eval_outputs)`.
        `outputs` is a tuple `(loss_outputs, evaluate_outputs)` which are
        averaged over all samples in `data`
      states: the initial states of learner that contains model parameters
        and other states. `states[0]` is always the model parameter.
    """
    if test_model is None:
        test_model = train_model
    if evaluator is None:
        evaluator = (lambda inputs, net_outputs, states: None, None)
    update = optimizer[0]
    backward = train_model[1]
    forward = test_model[0]
    evaluate = evaluator[0]
    initial_states = (train_model[2], optimizer[1], train_model[3],
                      test_model[3], evaluator[1])
    def train(data, states, callback=None):
        params = states[0]
        opt_states, model_states, eval_states = states[1], states[2], states[4]
        total_outputs = None
        for inputs in data:
            step = opt_states[0]
            grads, (net_outputs, loss_outputs), model_states = backward(
                params, inputs, model_states)
            eval_outputs, eval_states = evaluate(
                inputs, net_outputs, eval_states)
            outputs = (loss_outputs, eval_outputs)
            if total_outputs is None:
                total_outputs = outputs
            else:
                total_outputs = jax.tree_map(
                    lambda x, y: step / (step + 1) * x + 1 / (step + 1) * y,
                    total_outputs, outputs)
            params, opt_states = update(params, grads, states)

        states = (params, opt_states, model_states, states[3], eval_states)
        return outputs, states
    def test(data, states, callback=None):
        for inputs in data:
            pass
    return LearnerTuple(train, test, initial_states)


def dump(fd, states):
    """Save learner to file."""
    pass


def load(fd):
    """Load leaner from file."""
    pass
