"""
Functional neural network library for JAX.

Design principle: an xnn module is a function that returns 3 objects:
  forward: the forward function.
  params: initial module parameters; should be a pytree.
  states: initial states that forward will modify; should be a dict of pytrees (
    see below for meaning of the dict keys)
initial_params and initial_states can be None if not needed.

The signature of the forward function should be
outputs, states = forward(params, inputs, states)
You should use the returned new states for the next call to forward.

The states of a module consistitude a dict in which the keys control how to
split and aggregate states when vectorizing a module using xnn.vmap or xnn.pmap.
`states = {'rng': rng_states, 'mean': mean_states, k1: v1, k2: v2, ...}`
`rng_states` will be split to batch using jax.random.split before `forward`.
`mean_states` will be copied to batch before `forward` and averaged after.
States of all other keys will be copied before `forward` without any
postprocessing after.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple
import math

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.nn.initializers as jinit
import jax.random as jrand
import jax.tree_util as jtree
import jax.lax as jlax
import jax.image as jimage
from xjax import xrand


ModuleTuple = namedtuple('Module', ['forward', 'params', 'states'])


def Linear(in_dim, out_dim, w_init=jinit.glorot_normal(), b_init=jinit.normal(),
           rng=None):
    rng = rng if rng is not None else xrand.split()
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (in_dim, out_dim))
    initial_b = b_init(b_rng, (out_dim,))
    initial_params = (initial_w, initial_b)
    def forward(params, inputs, states):
        w, b = params
        return jnp.dot(inputs, w) + b, states
    return ModuleTuple(forward, initial_params, None)


def Embed(embed_size, embed_dim, embed_init=jinit.normal(), rng=None):
    rng = rng if rng is not None else xrand.split()
    initial_params = embed_init(rng, (embed_size, embed_dim))
    def forward(params, inputs, states):
        return jnp.take(params, inputs, axis=0), states
    return ModuleTuple(forward, initial_params, None)


def Dropout(p=0.5, mode='train', rng=None):
    rng = rng if rng is not None else xrand.split()
    def forward(params, inputs, states):
        if mode == 'train' and p != 0:
            new_rng, rng = jrand.split(states['rng'])
            keep = jrand.bernoulli(rng, 1 - p, inputs.shape)
            outputs = jnp.where(keep, inputs / (1 - p), 0)
            return outputs, {'rng': new_rng}
        else:
            return inputs, states
    return ModuleTuple(forward, None, {'rng': rng})


def Conv(in_dim, out_dim, kernel, stride=None, dilation=None,
         padding='SAME', w_init=jinit.glorot_normal(1, 0),
         b_init=jinit.normal(1e-6), rng=None, *args, **kwargs):
    """n-D convolutional layer.

    Args:
      in_dim: input feature dimension.
      out_dim: output feature dimension.
      kernel: a tuple of n integers representing convolution kernel size.
      stride: None or a tuple of n integers representing convolution strides.
      dilation: None or a tuple of n integers representing dilation.
      padding: string 'SAME', 'VALID' or a tuple of n (low, high) integer pairs
        that gives the padding to apply before and after each spatial dimension.
      w_init: initializer of kernel weights.
      b_init: initializer of bias vector.
      rng: random key. Use xrand.split() if None.

    Returns:
      forward: the forward function `outputs, states = forward(params, inputs,
        states)`. `inputs` should be a rank n + 1 input array in which the first
        dimension is the feature or channel of size in_dim.
      params: initial parameters for the convolutional layer.
      states: initial states for the convolutional layer.
    """
    rng = rng if rng is not None else xrand.split()
    stride = stride if stride is not None else (1,) * len(kernel)
    dilation = dilation if dilation is not None else (1,) * len(kernel)
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (out_dim, in_dim) + kernel)
    initial_b = b_init(b_rng, (out_dim,) + (1,) * len(kernel))
    initial_params = (initial_w, initial_b)
    def forward(params, inputs, states):
        w, b = params
        batch_mode = (inputs.ndim >= w.ndim)
        if not batch_mode:
            inputs = jnp.expand_dims(inputs, 0)
        batch_ndim = inputs.ndim - w.ndim + 1
        batch_dims = inputs.shape[0:batch_ndim]
        inputs = jnp.reshape(
            inputs, (math.prod(batch_dims),) + inputs.shape[batch_ndim:])
        outputs = jlax.conv_general_dilated(
            inputs, w, stride, padding, None, dilation, *args, **kwargs) + b
        outputs = jnp.reshape(
            outputs, batch_dims + outputs.shape[1:])
        if not batch_mode:
            outputs = jnp.squeeze(outputs, 0)
        return outputs, states
    return ModuleTuple(forward, initial_params, None)


def Deconv(in_dim, out_dim, kernel, stride=None, dilation=None, padding='SAME',
           w_init=jinit.glorot_normal(1, 0), b_init=jinit.normal(1e-6),
           rng=None, *args, **kwargs):
    """n-D deconvolutional (or transposed convolutional) layer. The parameters
    stride, dilation, and padding all correspond to a Conv layer and will
    be transposed for deconvolution.

    Args:
      in_dim: input feature dimension.
      out_dim: output feature dimension.
      kernel: a tuple of n integers representing convolution kernel size.
      stride: None or a tuple of n integers representing convolution strides.
      dilation: None or a tuple of n integers representing dilation.
      padding: string 'SAME', 'VALID' or a tuple of n (low, high) integer pairs
        that gives the convolution padding.
      w_init: initializer of kernel weights.
      b_init: initializer of bias vector.

    Returns:
      forward: the forward function `outputs, states = forward(params, inputs,
        states)`. `inputs` should be a rank n + 1 input array in which the first
        dimension is the feature or channel of size in_dim.
      params: initial parameters for the convolutional layer.
      states: initial states for the convolutional layer.
    """
    rng = rng if rng is not None else xrand.split()
    dimension = jlax.ConvDimensionNumbers(
        tuple(range(len(kernel) + 2)), tuple(range(len(kernel) + 2)),
        tuple(range(len(kernel) + 2)))
    stride = stride if stride is not None else (1,) * len(kernel)
    dilation = dilation if dilation is not None else (1,) * len(kernel)
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (out_dim, in_dim) + kernel)
    initial_b = b_init(b_rng, (out_dim,) + (1,) * len(kernel))
    initial_params = (initial_w, initial_b)
    def forward(params, inputs, states):
        w, b = params
        batch_mode = (inputs.ndim >= w.ndim)
        if not batch_mode:
            inputs = jnp.expand_dims(inputs, 0)
        batch_ndim = inputs.ndim - w.ndim + 1
        batch_dims = inputs.shape[0:batch_ndim]
        inputs = jnp.reshape(
            inputs, (math.prod(batch_dims),) + inputs.shape[batch_ndim:])
        outputs = jlax.conv_transpose(
            inputs, w, stride, padding, dilation, dimension,
            *args, **kwargs) + b
        outputs = jnp.reshape(
            outputs, batch_dims + outputs.shape[1:])
        if not batch_mode:
            outputs = jnp.squeeze(outputs, 0)
        return outputs, states
    return ModuleTuple(forward, initial_params, None)


def MaxPool(kernel, stride, dilation, padding='SAME', *args, **kwargs):
    """n-D max pooling layer.

    Args:
      kernel: a tuple of n integers representing the pooling kernel size.
      stride: None or a tuple of n integers representing pooling strides.
      dilation: None or a tuple of n integers representing dilation.
      padding: string 'SAME', 'VALID', or a tuple of n (low, high) integer pairs
        that gives the padding to apply before and after each spatial dimension.

    Return:
      forward: the forward function `outputs, states = forward(params, inputs,
        states)`. `inputs` should be a rank n + 1 input array in which the first
        dimension is the feature or channel.
      params: initial parameters for the pooling layer.
      states: initial states for the pooling layer.
    """
    stride = stride if stride is not None else kernel
    dilation = dilation if dilation is not None else (1,) * len(kernel)
    def forward(params, inputs, states):
        extra_ndim = inputs.ndim - len(kernel)
        _kernel = (1,) * extra_ndim + kernel
        _stride = (1,) * extra_ndim + stride
        _dilation = (1,) * extra_ndim + dilation
        outputs = jlax.reduce_window(
            inputs, -jnp.inf, jlax.max, _kernel, _stride, padding, *args,
            window_dilation=_dilation, **kwargs)
        return outputs, states
    return ModuleTuple(forward, None, None)


def AvgPool(kernel, stride, dilation, padding='SAME', *args, **kwargs):
    """n-D average pooling layer.

    Args:
      kernel: a tuple of n integers representing the pooling kernel size.
      stride: None or a tuple of n integers representing pooling strides.
      dilation: None or a tuple of n integers representing dilation.
      padding: string 'SAME', 'VALID', or a tuple of n (low, high) integer pairs
        that gives the padding to apply before and after each spatial dimension.

    Returns:
      forward: the forward function `outputs, states = forward(params, inputs,
        states)`. `inputs` should be a rank n + 1 input array in which the first
        dimension is the feature or channel.
      params: initial parameters for the pooling layer.
      states: initial states for the pooling layer.
    """
    stride = stride if stride is not None else kernel
    dilation = dilation if dilation is not None else (1,) * len(kernel)
    def forward(params, inputs, states):
        extra_ndim = inputs.ndim - len(kernel)
        _kernel = (1,) * extra_ndim + kernel
        _stride = (1,) * extra_ndim + stride
        _dilation = (1,) * extra_ndim + dilation
        outputs = jlax.reduce_window(
            inputs / math.prod(_kernel), -jnp.inf, jlax.add, _kernel, _stride,
            padding, *args, window_dilation=_dilation, **kwargs)
        return outputs, states
    return ModuleTuple(forward, None, None)


def Resize(factor, method='nearest', *args, **kwargs):
    """Resize the input using jax.image.resize().
    
    Args:
      factor: the factor of resize. The shape of outputs is factor multiplying
        the shape of inputs, floored to the nearest integer.
      method: the resize method.

    Returns:
      forward: the forward function.
      params: initial parameteers.
      states: initial states.
    """
    def forward(params, inputs, states):
        shape = (math.floor(s * f) for s, f in zip(inputs.shape, factor))
        outputs = jimage.resize(inputs, shape, method, *args, **kwargs)
        return outputs, states
    return ModuleTuple(forward, None, None)


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
Square = partial(SingleInput, jnp.square)
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


def Random(func, rng=None, *args, **kwargs):
    """Layer that generate random numbers."""
    rng = rng if rng is not None else xrand.split()
    def forward(params, inputs, states):
        func_rng, new_rng = jrand.split(states['rng'])
        return func(func_rng, *args, **kwargs), {'rng': new_rng}
    return ModuleTuple(forward, None, {'rng': rng})
Normal = partial(Random, jrand.normal)
Uniform = partial(Random, jrand.uniform)
Bernoulli = partial(Random, jrand.bernoulli)


def pack_states(states):
    """Pack states for container."""
    new_states = {}
    for i in (states if states is not None else {}):
        for key in (states[i] if states[i] is not None else {}):
            if key not in new_states:
                new_states[key] = {i: states[i][key]}
            else:
                new_states[key][i] = states[i][key]
    if len(new_states) == 0:
        new_states = None
    return new_states

def pack_states_list(states):
    """Pack states list for container."""
    new_states = {}
    if states is not None:
        for i in range(len(states)):
            new_states[i] = states[i]
    if len(new_states) == 0:
        new_states = None
    return pack_states(new_states)

def unpack_states(states):
    """Unpack states for container."""
    new_states = {}
    for key in (states if states is not None else {}):
        for i in (states[key] if states[key] is not None else {}):
            if i not in new_states:
                new_states[i] = {key: states[key][i]}
            else:
                new_states[i][key] = states[key][i]
    if len(new_states) == 0:
        new_states = None
    return new_states


def Sequential(*modules):
    """Sequential container."""
    forwards, initial_params, initial_states_list = zip(*modules)
    initial_states = pack_states_list(initial_states_list)
    def forward(params, inputs, states):
        outputs = inputs
        states = unpack_states(states)
        new_states = {}
        for i in range(len(forwards)):
            if states is not None and i in states:
                outputs, new_states[i] = forwards[i](
                    params[i], outputs, states[i])
            else:
                outputs, _ = forwards[i](params[i], outputs, None)
        return outputs, pack_states(new_states)
    return ModuleTuple(forward, initial_params, initial_states)


def Parallel(*modules):
    """Parallel container."""
    forwards, initial_params, initial_states_list = zip(*modules)
    initial_states = pack_states_list(initial_states_list)
    def forward(params, inputs, states):
        states = unpack_states(states)
        outputs, new_states = [], {}
        for i in range(len(forwards)):
            if states is not None and i in states:
                outputs_i, new_states[i] = forwards[i](
                    params[i], inputs[i], states[i])
            else:
                outputs_i, _ = forwards[i](params[i], inputs[i], None)
            outputs.append(outputs_i)
        outputs = type(inputs)(outputs)
        return outputs, pack_states(new_states)
    return ModuleTuple(forward, initial_params, initial_states)


def SharedParallel(module):
    """Share module parameters across multiple parallel inputs."""
    module_forward, initial_params, initial_states = module
    def forward(params, inputs, states):
        outputs = [None,]*len(inputs)
        for i in range(len(inputs)):
            outputs[i], states = module_forward(params, inputs[i], states)
        outputs = type(inputs)(outputs)
        return outputs, states
    return ModuleTuple(forward, initial_params, initial_states)


def copy_vectorize(states, size):
    """vectorize states before calling forward."""
    return jax.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), size, axis=0), states)

def rng_vectorize(states, size):
    """vectorize rng states before calling forward."""
    def leaf_vectorize(leaf):
        new_leaf = jnp.apply_along_axis(
            lambda rng: jnp.array(jrand.split(rng, size)), -1, leaf)
        return jnp.swapaxes(new_leaf, -2, 0)
    new_states = jax.tree_map(leaf_vectorize, states)
    return new_states

def vectorize_states(states, size):
    new_states = None
    if states is not None:
        new_states = {}
        for key in states:
            if key == 'rng':
                new_states[key] = rng_vectorize(states[key], size)
            else:
                new_states[key] = copy_vectorize(states[key], size)
    return new_states

def mean_postprocess(states, size):
    """vectorize sum states after calling forward."""
    def leaf_vectorize(leaf):
        return jax.repeat(jax.mean(leaf, axis=0, keepdims=True), size, axis=0)
    new_states = jax.tree_map(leaf_vectorize, states)

def postprocess_states(states, size):
    new_states = None
    if states is not None:
        new_states = {}
        for key in states:
            if key == 'mean':
                new_states[key] = mean_postprocess(states[key], size)
            else:
                new_states[key] = states[key]
    return new_states

def vectorize(map_func, module, size, *args, **kwargs):
    """Vectorize the module.

    Args:
      map_func: jax.vmap or jax.pmap.
      module: the module to be vectorized.
      size: the batch size.

    Returns:
      forward: the vectorized forward function.
      params: module parameters.
      states: vectorized states according to its dictionary key.
    """
    module_forward, initial_params, module_states = module
    initial_states = vectorize_states(module_states, size)
    # Map over inputs and states, but not parameters.
    forward_v = map_func(module_forward, in_axes=(None, 0, 0), *args, **kwargs)
    def forward(params, inputs, states):
        outputs, states = forward_v(params, inputs, states)
        new_states = postprocess_states(states, size)
        return outputs, new_states
    return ModuleTuple(forward, initial_params, initial_states)

vmap = partial(vectorize, jax.vmap)
pmap = partial(vectorize, jax.pmap)
