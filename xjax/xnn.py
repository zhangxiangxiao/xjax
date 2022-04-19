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

from functools import partial

import jax.numpy as jnp
import jax.nn as jnn
import jax.nn.initializers as jinit
import jax.random as jrand


def Linear(rng, in_dim, out_dim, w_init=jinit.glorot_normal(),
           b_init=jinit.normal()):
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (out_dim, in_dim))
    initial_b = b_init(b_rng, (out_dim,))
    initial_params = (initial_w, initial_b)
    def forward(params, inputs, states):
        w, b = params
        return jnp.dot(w, inputs) + b, states
    return forward, initial_params, None


def Embed(rng, embed_size, embed_dim, embed_init=jinit.normal()):
    initial_params = embed_init(rng, (embed_size, embed_dim))
    def forward(params, inputs, states):
        return jnp.take(params, inputs, axis=0), states
    return forward, initial_params, None


def Dropout(rng, p=0.5, mode='train'):
    initial_states = rng
    def forward(params, inputs, states):
        """`states` is actually an rng."""
        if mode == 'train':
            new_states, rng = jrand.split(states)
            keep = jrand.bernoulli(rng, p, inputs.shape)
            outputs = jnp.where(keep, inputs / p, 0)
        else:
            new_states = states
            outputs = inputs
        return outputs, new_states
    return forward, None, initial_states


def SingleInputFunc(func, **kwargs):
    """Layer that feed func with inputs.
    Used for modules that do not have params and states. Hyper-parameters are
    stored in kwargs as a Python3 function closure."""
    def forward(params, inputs, states):
        return func(inputs, **kwargs), states
    return forward, None, None
# Transfer functions
Abs = partial(SingleInputFunc, jnp.abs)
Tanh = partial(SingleInputFunc, jnp.tanh)
Exp = partial(SingleInputFunc, jnp.exp)
Relu = partial(SingleInputFunc, jnn.relu)
Sigmoid = partial(SingleInputFunc, jnn.sigmoid)
Softplus = partial(SingleInputFunc, jnn.softplus)
LogSigmoid = partial(SingleInputFunc, jnn.log_sigmoid)
Softmax = partial(SingleInputFunc, jnn.softmax)
LogSoftmax = partial(SingleInputFunc, jnn.log_softmax)
Standardize = partial(SingleInputFunc, jnn.standardize)
# Reduction functions
Max = partial(SingleInputFunc, jnp.max)
Mean = partial(SingleInputFunc, jnp.mean)
Median = partial(SingleInputFunc, jnp.median)
Min = partial(SingleInputFunc, jnp.min)
Prod = partial(SingleInputFunc, jnp.prod)
Std = partial(SingleInputFunc, jnp.std)
Sum = partial(SingleInputFunc, jnp.sum)
Var = partial(SingleInputFunc, jnp.var)
Norm = partial(SingleInputFunc, jnp.linalg.norm)
Logsumexp = partial(SingleInputFunc, jnn.logsumexp)
# Shape transformation functions
Transpose = partial(SingleInputFunc, jnp.transpose)
Reshape = partial(SingleInputFunc, jnp.reshape)
Repeat = partial(SingleInputFunc, jnp.repeat)


def mul_const(inputs, const):
    """Multiply by a constant."""
    return inputs * const
MulConst = partial(SingleInputFunc, mul_const)


def add_const(inputs, const):
    """Add a constant."""
    return inputs + const
AddConst = partial(SingleInputFunc, add_const)


def group(inputs, indices):
    """Group inputs into multipe sub-groups.
    Example: group(inputs, ([0,2,4],[1,3])) for a list inputs object will return
    a tuple ([inputs[0], inputs[2], inputs[4]], [inputs[1], inputs[3]]).
    type(inputs) determines the type (list or tuple) of the subgroup.
    type(indices) determines the type (list or tuple) of the group container.
    """
    outputs = type(indices)(type(inputs)(inputs[i] for i in sub_indices)
               for sub_indices in indices)
    return outputs
Group = partial(SingleInputFunc, group)


def ungroup(inputs):
    """Ungroup container of containers.
    Example: ungroup([[[0,1],2],[3,4]]) = [[0,1],2,3,4]. [0,1] is not flattened
    because it is deeper than 2 containers.
    type(inputs) determines the type (list or tuple) of the ungrouped outputs.
    """
    outputs = type(inputs)(item for sub_inputs in inputs for item in sub_inputs)
    return outputs
Ungroup = partial(SingleInputFunc, ungroup)


def unpack(inputs):
    """Unpack inputs by assuming it has only one member: `outputs, = inputs.`"""
    outputs, = inputs
    return outputs
Unpack = partial(SingleInputFunc, unpack)


def MultiInputFunc(func, **kwargs):
    """Layer that applies func with inputs unpacked.
    Used for modules that accept multiple inputs and do not have params or
    states. Hyper-parameters are stored in kwargs as a Python3 function
    closure."""
    def forward(params, inputs, states):
        return func(*inputs, **kwargs), states
    return forward, None, None
# Arithmetic functions
Add = partial(MultiInputFunc, jnp.add)
Subtract = partial(MultiInputFunc, jnp.subtract)
Multiply = partial(MultiInputFunc, jnp.multiply)
Divide = partial(MultiInputFunc, jnp.divide)
# Linear algebra functions
MatMul = partial(MultiInputFunc, jnp.matmul)
Dot = partial(MultiInputFunc, jnp.dot)


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
    return forward, initial_params, initial_states


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
    return forward, initial_params, initial_states


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
    return forward, initial_params, initial_states
