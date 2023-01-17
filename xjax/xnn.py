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
synchronize states when vectorizing a module using xnn.vectorize.
`states = {'rng': rng_states, 'mean': mean_states, k1: v1, k2: v2, ...}`
`rng_states` will be split to batch using jax.random.split before `forward` and
aggregated after by selecting the rng of the first sample.
`mean_states` will be copied to batch before `forward` and averaged after.
States of all other keys will be copied before `forward` and aggregated by
selecting the state outputs of the first sample.
"""

from __future__ import absolute_import
from collections import namedtuple
from functools import partial, wraps
import math

import jax
import jax.image as jimage
import jax.lax as jlax
import jax.nn as jnn
import jax.nn.initializers as jinit
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtree
import numpy as np
from xjax import xrand


ModuleTuple = namedtuple('Module', ['forward', 'params', 'states'])


def tree_forward(forward):
    """Make forward function work with pytrees."""
    @wraps(forward)
    def wrapped_forward(params, inputs, states):
        flat_inputs, treedef = jtree.tree_flatten(inputs)
        flat_outputs = []
        for leaf_inputs in flat_inputs:
            leaf_outputs, states = forward(params, leaf_inputs, states)
            flat_outputs.append(leaf_outputs)
        outputs = treedef.unflatten(flat_outputs)
        return outputs, states
    return wrapped_forward


def Linear(in_dim, out_dim, w_init=jinit.glorot_normal(), b_init=jinit.normal(),
           rng=None):
    rng = rng if rng is not None else xrand.split()
    w_rng, b_rng = jrand.split(rng)
    initial_w = w_init(w_rng, (out_dim, in_dim))
    initial_b = b_init(b_rng, (out_dim,))
    initial_params = (initial_w, initial_b)
    @tree_forward
    def forward(params, inputs, states):
        w, b = params
        return jnp.dot(inputs, jnp.transpose(w)) + b, states
    return ModuleTuple(forward, initial_params, None)


def Embed(embed_size, embed_dim, embed_init=jinit.glorot_normal(), rng=None):
    rng = rng if rng is not None else xrand.split()
    initial_params = embed_init(rng, (embed_size, embed_dim))
    @tree_forward
    def forward(params, inputs, states):
        return jnp.take(params, inputs, axis=0), states
    return ModuleTuple(forward, initial_params, None)


def Dropout(p=0.5, mode='train', rng=None):
    rng = rng if rng is not None else xrand.split()
    @tree_forward
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
    @tree_forward
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
    @tree_forward
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


def MaxPool(kernel, stride=None, dilation=None, padding='SAME',
            *args, **kwargs):
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
    @tree_forward
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


def AvgPool(kernel, stride=None, dilation=None, padding='SAME', *args,
            **kwargs):
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
    @tree_forward
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


def Resize(shape, method='nearest', *args, **kwargs):
    """Resize the inputs using jax.image.resize().

    Args:
      shape: the shape to resize inputs into.
      method: the resize method.

    Returns:
      forward: the forward function.
      params: initial parameters.
      states: initial states.
    """
    @tree_forward
    def forward(params, inputs, states):
        outputs = jimage.resize(inputs, shape, method, *args, **kwargs)
        return outputs, states
    return ModuleTuple(forward, None, None)


def ResizeLike(method='nearest', *args, **kwargs):
    """Resize inputs[0] to the shape of inputs[1]. Both are returned.

    Args:
      method: the resize method.

    Returns:
      forward: the forward function.
      params: initial parameters.
      states: initial_states.
    """
    def forward(params, inputs, states):
        def leaf_resize(inputs0, inputs1):
            return jimage.resize(inputs0, inputs1.shape, method,
                                 *args, **kwargs)
        outputs = [jax.tree_map(leaf_resize, inputs[0], inputs[1]), inputs[1]]
        return type(inputs)(outputs), states
    return ModuleTuple(forward, None, None)


def FlattenUpTo(tree):
    """Flattens inputs according to the structure of tree.

    Args:
      tree: a tree representing the tree structure.

    Returns:
      forward: the forward function.
      params: initial parameteers.
      states: initial states.
    """
    treedef = jtree.tree_structure(tree)
    def forward(params, inputs, states):
        return treedef.flatten_up_to(inputs), states
    return ModuleTuple(forward, None, None)


def Identity():
    def forward(params, inputs, states):
        return inputs, states
    return ModuleTuple(forward, None, None)


def Group(ind):
    """Group inputs into a tree structure according to ind.
    Example: Group([1, [0, 2]]) will return [inputs[1], [inputs[0], inputs[2]]].
    """
    def forward(params, inputs, states):
        outputs = jax.tree_map(lambda x: inputs[x], ind)
        return outputs, states
    return ModuleTuple(forward, None, None)


def Flatten():
    """Flatten inputs into a list."""
    def forward(params, inputs, states):
        outputs, _ = jtree.tree_flatten(inputs)
        return outputs, states
    return ModuleTuple(forward, None, None)


def Pack():
    """Pack inputs by returnining [inputs]."""
    def forward(params, inputs, states):
        outputs = [inputs]
        return outputs, states
    return ModuleTuple(forward, None, None)


def Unpack():
    """Unpack inputs by assuming it has only one member: `outputs, = inputs.`"""
    def forward(params, inputs, states):
        outputs, = inputs
        return outputs, states
    return ModuleTuple(forward, None, None)


def Reverse():
    """Reverse inputs."""
    def forward(params, inputs, states):
        return inputs[::-1], states
    return ModuleTuple(forward, None, None)


def SingleInput(func, *args, **kwargs):
    """Layer that feed func with inputs. Used for modules that do not have
    params and states. Hyper-parameters are stored in kwargs as a Python3
    function closure."""
    @tree_forward
    def forward(params, inputs, states):
        return func(inputs, *args, **kwargs), states
    return ModuleTuple(forward, None, None)
# Transfer functions
Abs = partial(SingleInput, jnp.abs)
Tanh = partial(SingleInput, jnp.tanh)
Exp = partial(SingleInput, jnp.exp)
Log = partial(SingleInput, jnp.log)
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
# Construction functions
ZerosLike = partial(SingleInput, jnp.zeros_like)
OnesLike = partial(SingleInput, jnp.ones_like)
FullLike = partial(SingleInput, jnp.full_like)
# Shape transformation functions
Transpose = partial(SingleInput, jnp.transpose)
Reshape = partial(SingleInput, jnp.reshape)
Repeat = partial(SingleInput, jnp.repeat)
Split = partial(SingleInput, jnp.split)
# Others
OneHot = partial(SingleInput, jnn.one_hot)
Pad = partial(SingleInput, jnp.pad)

def mul_const(inputs, const):
    """Multiply by a constant."""
    return inputs * const
MulConst = partial(SingleInput, mul_const)

def add_const(inputs, const):
    """Add a constant."""
    return inputs + const
AddConst = partial(SingleInput, add_const)

def logcosh(inputs):
    return jnp.logaddexp(inputs, -inputs) - math.log(2)
LogCosh = partial(SingleInput, logcosh)

def depad(array, pad_width):
    """Transpose of pad."""
    pad_width = np.asarray(pad_width)
    if not pad_width.dtype.kind == 'i':
        raise TypeError('pad_width must be of integral type.')
    pad_width = np.broadcast_to(pad_width, (array.ndim, 2)).tolist()
    slices = tuple(slice(p[0], None if p[1] == 0 else -p[1]) for p in pad_width)
    return array[slices]
Depad = partial(SingleInput, depad)


def MultiInput(func, *args, **kwargs):
    """ Layer that applies func with inputs unpacked. Used for modules that
    accept multiple inputs and do not have params or states. Hyper-parameters
    are stored in kwargs as a function closure.
    """
    def forward(params, inputs, states):
        def wrapped_func(*inputs):
            return func(*inputs, *args, **kwargs)
        return jax.tree_map(wrapped_func, *inputs), states
    return ModuleTuple(forward, None, None)
# Arithmetic functions
Add = partial(MultiInput, jnp.add)
Subtract = partial(MultiInput, jnp.subtract)
Multiply = partial(MultiInput, jnp.multiply)
Divide = partial(MultiInput, jnp.divide)
Logaddexp = partial(MultiInput, jnp.logaddexp)
Power = partial(MultiInput, jnp.power)
# Linear algebra functions
Matmul = partial(MultiInput, jnp.matmul)
Dot = partial(MultiInput, jnp.dot)

def blse(x, p, axis=None):
    """Binary log-sum-exp loss. When `axis == None`, it operates in binary mode
    blse(x,p) = ln(1+p^2*exp(-x)+(1-p)^2*exp(x)) - ln(1+2p(1-p)),
    when `axis != None`, it calcuates in categorical mode
    blse(x,p) = ln(1+sum_i[p_i^2*exp(-x_i)]+sum_i[(1-p_i)^2*exp(x_i)]) -
                ln(1+sum_i[2p_i(1-p_i)])

    Args:
      x: the unnormalized logit.
      p: the target probability.
      axis: the categorical axis. If `None`, will operate in binary mode.

    Returns:
      outputs: the output of the binary log-sum-exp loss.
    """
    a = jnp.stack((-x, x), axis=-1)
    b_sqrt = jnp.stack((p, 1-p), axis=-1)
    blse_pos = jnn.logsumexp(a=a, axis=-1, b=jnp.square(b_sqrt))
    blse_neg = 2 * jnp.prod(b_sqrt, axis=-1)
    if axis is not None:
        blse_pos = jnn.logsumexp(blse_pos, axis=axis)
        blse_neg = jnp.sum(blse_neg, axis=axis)
    return jnn.softplus(blse_pos) - jnp.log1p(blse_neg)
BLSE = partial(MultiInput, blse)


def ListInput(func, *args, **kwargs):
    """ Layer that applies func to an input which is a list. Hypaer-parameters
    are stored in kwargs as a function closure.
    """
    def forward(params, inputs, states):
        def wrapped_func(*inputs):
            return func(inputs, *args, **kwargs)
        return jax.tree_map(wrapped_func, *inputs), states
    return ModuleTuple(forward, None, None)
Concatenate = partial(ListInput, jnp.concatenate)
Stack = partial(ListInput, jnp.stack)


def RandomLike(func, rng=None, *args, **kwargs):
    """Layer that generate random numbers."""
    rng = rng if rng is not None else xrand.split()
    @tree_forward
    def forward(params, inputs, states):
        func_rng, new_rng = jrand.split(states['rng'])
        outputs = func(func_rng, *args, shape=inputs.shape, **kwargs)
        return outputs, {'rng': new_rng}
    return ModuleTuple(forward, None, {'rng': rng})
NormalLike = partial(RandomLike, jrand.normal)
UniformLike = partial(RandomLike, jrand.uniform)
BernoulliLike = partial(RandomLike, jrand.bernoulli)
ExponentialLike = partial(RandomLike, jrand.exponential)
RandintLike = partial(RandomLike, jrand.randint)


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


def DenseSequential(*modules):
    """Sequential where outputs are the concatenated outputs of all layers."""
    forwards, initial_params, initial_states_list = zip(*modules)
    initial_states = pack_states_list(initial_states_list)
    def forward(params, inputs, states):
        outputs = []
        states = unpack_states(states)
        module_outputs = inputs
        new_states = {}
        for i in range(len(forwards)):
            if states is not None and i in states:
                module_outputs, new_states[i] = forwards[i](
                    params[i], module_outputs, states[i])
            else:
                module_outputs, _ = forwards[i](params[i], module_outputs, None)
            outputs.append(module_outputs)
        return outputs, pack_states(new_states)
    return ModuleTuple(forward, initial_params, initial_states)


def MergeSequential(*modules):
    """Sequential where inputs are merged into the intermediate outputs."""
    forwards, initial_params, initial_states_list = zip(*modules)
    initial_states = pack_states_list(initial_states_list)
    def forward(params, inputs, states):
        states = unpack_states(states)
        outputs = None
        new_states = {}
        for i in range(len(forwards)):
            if outputs == None:
                module_inputs = inputs[i]
            else:
                module_inputs = [inputs[i], outputs]
            if states is not None and i in states:
                outputs, new_states[i] = forwards[i](
                    params[i], module_inputs, states[i])
            else:
                outputs, _ = forwards[i](params[i], module_inputs, None)
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


def copy_vectorize(states, batch):
    """vectorize states before calling forward."""
    return jax.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch, axis=0), states)

def rng_vectorize(states, batch):
    """vectorize rng states before calling forward."""
    def leaf_vectorize(leaf):
        new_leaf = jnp.apply_along_axis(
            lambda rng: jnp.array(jrand.split(rng, batch)), -1, leaf)
        return jnp.swapaxes(new_leaf, -2, 0)
    new_states = jax.tree_map(leaf_vectorize, states)
    return new_states

def vectorize_states(states, batch):
    new_states = None
    if states is not None:
        new_states = {}
        for key in states:
            if key == 'rng':
                new_states[key] = rng_vectorize(states[key], batch)
            else:
                new_states[key] = copy_vectorize(states[key], batch)
    return new_states

def copy_unvectorize(states):
    """Unvectorize the copied states."""
    # Simply choose the first rng.
    return jax.tree_map(lambda x: x[0], states)

def mean_unvectorize(states):
    """Unvectorize the mean'ed states."""
    def leaf_unvectorize(leaf):
        return jax.mean(leaf, axis=0, keepdims=False)
    return jax.tree_map(leaf_unvectorize, states)

def unvectorize_states(states):
    new_states = None
    if states is not None:
        new_states = {}
        for key in states:
            if key == 'mean':
                new_states[key] = mean_unvectorize(states[key])
            else:
                new_states[key] = copy_unvectorize(states[key])
    return new_states


def vectorize(module, map_func=jax.vmap, *args, **kwargs):
    """Vectorize the module.

    Args:
      module: the module to be vectorized.
      map_func: jax.vmap or jax.pmap.

    Returns:
      forward: the vectorized forward function.
      params: module parameters.
      states: vectorized states according to its dictionary key.
    """
    module_forward, initial_params, initial_states = module
    # Map over inputs and states, but not parameters.
    forward_v = map_func(module_forward, in_axes=(None, 0, 0), *args, **kwargs)
    def forward(params, inputs, states):
        batch = jtree.tree_leaves(inputs)[0].shape[0]
        states = vectorize_states(states, batch)
        outputs, states = forward_v(params, inputs, states)
        states = unvectorize_states(states)
        return outputs, states
    return ModuleTuple(forward, initial_params, initial_states)


def jit(module, *args, **kwargs):
    """Set up the module for JIT.

    Args:
      module: an xnn module.

    Returns:
      jit_module: JIT'ed module.
    """
    forward, params, states = module
    return ModuleTuple(jax.jit(forward, *args, **kwargs), params, states)
