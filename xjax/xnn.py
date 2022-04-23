"""
Functional neural network library for JAX.

Design principle: an xjax module is a function that returns 3 objects:
  forward: the forward function.
  initial_params: initial module parameters; should be a pytree.
  initial_states: initial states that forward will modify; should be picklable.
initial_params and initial_states can be None if not needed.

The signature of the forward function should be
outputs, new_states = forward(params, inputs, states)
You should use the returned new states for the next call to forward.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.nn.initializers as jinit
import jax.random as jrand
import jax.tree_util as jtree


ModuleTuple = namedtuple('Module', ['forward', 'params', 'states'])


def Linear(rng, in_dim, out_dim, w_init=jinit.glorot_normal(),
           b_init=jinit.normal()):
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (out_dim, in_dim))
    initial_b = b_init(b_rng, (out_dim,))
    initial_params = (initial_w, initial_b)
    def forward(params, inputs, states):
        w, b = params
        return jnp.dot(w, inputs) + b, states
    return ModuleTuple(forward, initial_params, None)


def Embed(rng, embed_size, embed_dim, embed_init=jinit.normal()):
    initial_params = embed_init(rng, (embed_size, embed_dim))
    def forward(params, inputs, states):
        return jnp.take(params, inputs, axis=0), states
    return ModuleTuple(forward, initial_params, None)


def Dropout(rng, p=0.5, mode='train'):
    def forward(params, inputs, states):
        if mode == 'train' and p != 0:
            """`states` is actually an rng."""
            new_states, rng = jrand.split(states)
            keep = jrand.bernoulli(rng, 1 - p, inputs.shape)
            outputs = jnp.where(keep, inputs / (1 - p), 0)
            return outputs, new_states
        else:
            return inputs, states
    return ModuleTuple(forward, None, rng)


def SingleInput(func, *args, **kwargs):
    """Layer that feed func with inputs.
    Used for modules that do not have params and states. Hyper-parameters are
    stored in kwargs as a Python3 function closure."""
    def forward(params, inputs, states):
        return func(inputs, *args, **kwargs), states
    return ModuleTuple(forward, None, None)
# Transfer functions
Abs = partial(SingleInput, jnp.abs)
Tanh = partial(SingleInput, jnp.tanh)
Exp = partial(SingleInput, jnp.exp)
ReLU = partial(SingleInput, jnn.relu)
Sigmoid = partial(SingleInput, jnn.sigmoid)
Softplus = partial(SingleInput, jnn.softplus)
LogSigmoid = partial(SingleInput, jnn.log_sigmoid)
Softmax = partial(SingleInput, jnn.softmax)
LogSoftmax = partial(SingleInput, jnn.log_softmax)
Standardize = partial(SingleInput, jnn.standardize)
# Reduction functions
Max = partial(SingleInput, jnp.max)
Mean = partial(SingleInput, jnp.mean)
Median = partial(SingleInput, jnp.median)
Min = partial(SingleInput, jnp.min)
Prod = partial(SingleInput, jnp.prod)
Std = partial(SingleInput, jnp.std)
Sum = partial(SingleInput, jnp.sum)
Var = partial(SingleInput, jnp.var)
Norm = partial(SingleInput, jnp.linalg.norm)
Logsumexp = partial(SingleInput, jnn.logsumexp)
# Shape transformation functions
Transpose = partial(SingleInput, jnp.transpose)
Reshape = partial(SingleInput, jnp.reshape)
Repeat = partial(SingleInput, jnp.repeat)


# Identity
def identity(inputs):
    return inputs
Identity = partial(SingleInput, identity)


def mul_const(inputs, const):
    """Multiply by a constant."""
    return inputs * const
MulConst = partial(SingleInput, mul_const)


def add_const(inputs, const):
    """Add a constant."""
    return inputs + const
AddConst = partial(SingleInput, add_const)


def group(inputs, ind):
    """Group inputs into a tree structure according to ind.
    Example: group(inputs, [1, [0, 2]]) will return
    [inputs[1], [inputs[0], inputs[2]]].
    """
    outputs = jax.tree_map(lambda x: inputs[x], ind)
    return outputs
Group = partial(SingleInput, group)


def flatten(inputs):
    """Flatten inputs into a list."""
    outputs, _ = jtree.tree_flatten(inputs)
    return outputs
Flatten = partial(SingleInput, flatten)


def unpack(inputs):
    """Unpack inputs by assuming it has only one member: `outputs, = inputs.`"""
    outputs, = inputs
    return outputs
Unpack = partial(SingleInput, unpack)


def MultiInput(func, *args, **kwargs):
    """Layer that applies func with inputs unpacked.
    Used for modules that accept multiple inputs and do not have params or
    states. Hyper-parameters are stored in kwargs as a Python3 function
    closure."""
    def forward(params, inputs, states):
        return func(*inputs, *args, **kwargs), states
    return ModuleTuple(forward, None, None)
# Arithmetic functions
Add = partial(MultiInput, jnp.add)
Subtract = partial(MultiInput, jnp.subtract)
Multiply = partial(MultiInput, jnp.multiply)
Divide = partial(MultiInput, jnp.divide)
# Linear algebra functions
MatMul = partial(MultiInput, jnp.matmul)
Dot = partial(MultiInput, jnp.dot)


def Random(func, rng, *args, **kwargs):
    """Layer that generate random numbers."""
    def forward(params, inputs, states):
        rng, states = jrand.split(states)
        return func(rng, *args, **kwargs), states
    return ModuleTuple(forward, None, rng)
Normal = partial(Random, jrand.normal)
Uniform = partial(Random, jrand.uniform)
Bernoulli = partial(Random, jrand.bernoulli)


def Sequential(*modules):
    """Sequential container."""
    forwards, initial_params, initial_states = zip(*modules)
    def forward(params, inputs, states):
        outputs = inputs
        new_states = [None,]*len(states)
        for i in range(len(forwards)):
            outputs, new_states[i] = forwards[i](params[i], outputs, states[i])
        new_states = type(states)(new_states)
        return outputs, new_states
    return ModuleTuple(forward, initial_params, initial_states)


def Parallel(*modules):
    """Parallel container."""
    forwards, initial_params, initial_states = zip(*modules)
    def forward(params, inputs, states):
        results = [forwards[i](
            params[i], inputs[i], states[i]) for i in range(len(inputs))]
        outputs, new_states = zip(*results)
        outputs = type(inputs)(outputs)
        new_states = type(states)(new_states)
        return outputs, new_states
    return ModuleTuple(forward, initial_params, initial_states)


def SharedParallel(module):
    """Share module parameters across multiple parallel inputs."""
    module_forward, initial_params, initial_states = module
    def forward(params, inputs, states):
        outputs = [None,]*len(inputs)
        new_states = states
        for i in range(len(inputs)):
            outputs[i], new_states = module_forward(
                params, inputs[i], new_states)
        outputs = type(inputs)(outputs)
        return outputs, new_states
    return ModuleTuple(forward, initial_params, initial_states)
