"""
Common models defined for JAX using xjax.xnn protocol.

A model is a container of paramterized modules and their losses. It provides a
'backward' function in addition to `forward`. A model function as the following
signature:
forward, backward, initial_params, initial_states = Model(*args, **kwargs)

The `forward` function has the following signature:
`net_outputs, loss_outputs, new_states = forward(params, inputs, states)'

The `backward` function has the following signature:
`grads, net_outputs, loss_outputs, new_states = backward(params, inputs, states)`
Parameter gradients are stored in `grads`.

The `states` variable contains a tuple of xnn module states.
"""

from __future__ import absolute_import

from functools import partial
from collections import namedtuple

import jax
from jax import numpy as jnp
from jax import tree_util as jtree
from xjax import xnn


ModelTuple = namedtuple(
    'ModelTuple', ['forward', 'backward', 'params', 'states'])


def vjp(forward, params, inputs, states):
    """jax.vjp against the params of forward."""
    def _forward(_params):
        return forward(_params, inputs, states)
    outputs, _vjpf, states = jax.vjp(_forward, params, has_aux=True)
    def vjpf(grad_outputs):
        grad_params, = _vjpf(grad_outputs)
        return grad_params
    return vjpf, outputs, states

def vjp_full(forward, params, inputs, states):
    """jax.vjp against params and inputs of forward."""
    def _forward(_params, _inputs):
        return forward(_params, _inputs, states)
    outputs, vjpf, states = jax.vjp(_forward, params, inputs, has_aux=True)
    return vjpf, outputs, states

def vjp_inputs(forward, params, inputs, states):
    """jax.vjp against the inputs of forward."""
    def _forward(_inputs):
        return forward(params, _inputs, states)
    outputs, _vjpf, states = jax.vjp(_forward, inputs, has_aux=True)
    def vjpf(grad_outputs):
        grad_inputs, = _vjpf(grad_outputs)
        return grad_inputs
    return vjpf, outputs, states


def map_ones_like(tree):
    return jax.tree_map(jnp.ones_like, tree)

def map_zeros_like(tree):
    return jax.tree_map(jnp.zeros_like, tree)

def map_add(tree1, tree2):
    return jax.tree_map(jnp.add, tree1, tree2)


def Model(net, loss):
    """A model assuming `net_inputs, net_targets = inputs`.

    Args:
      net: an xjax.xnn module that has learnable parameters.
      loss: an xjax.xnn module that is used as a loss.

    Returns:
      forward: the forward function that returns net_outputs, loss_outputs, and
        states.
      backward: the backward function that returns grads and forward returns.
      initial_params: the initial parameters from net.
      initial_states: the initial states.
    """
    net_forward, initial_params, net_initial_states = net
    loss_forward, loss_params, loss_initial_states = loss
    initial_states = (net_initial_states, loss_initial_states)
    def forward(params, inputs, states):
        net_states, loss_states = states
        net_inputs, net_targets = inputs
        net_outputs, net_states = net_forward(params, net_inputs, net_states)
        loss_inputs = [net_outputs, net_targets]
        loss_outputs, loss_states = loss_forward(
            loss_params, loss_inputs, loss_states)
        states = (net_states, loss_states)
        return net_outputs, loss_outputs, states
    def backward(params, inputs, states):
        # Forward propagate and build backward graph.
        net_states, loss_states = states
        net_inputs, net_targets = inputs
        net_vjpf, net_outputs, net_states = vjp(
            net_forward, params, net_inputs, net_states)
        loss_inputs = [net_outputs, net_targets]
        loss_vjpf, loss_outputs, loss_states = vjp_inputs(
            loss_forward, loss_params, loss_inputs, loss_states)
        states = (net_states, loss_states)
        # Backward propagate.
        grads_loss_outputs = map_ones_like(loss_outputs)
        grads_net_outputs, _ = loss_vjpf(grads_loss_outputs)
        grads = net_vjpf(grads_net_outputs)
        return grads, net_outputs, loss_outputs, states
    return ModelTuple(forward, backward, initial_params, initial_states)


def GAN(gen, disc, gen_loss, disc_loss):
    """Generative adversarial networks model.

    Args:
      gen: the generator module.
      disc: the discriminator module.
      gen_loss: the generator loss.
      disc_loss: the discriminator loss.

    Returns:
      forward: the forward function that returns net_outputs, loss_outputs, and
        states.
      backward: the backward function that returns grads and forward returns.
      initial_params: the initial parameters from net.
      initial_states: the initial states.
    """
    initial_params = (gen[1], disc[1])
    initial_states = (gen[2], disc[2], gen_loss[2], disc_loss[2])
    gen_forward, disc_forward = gen[0], disc[0]
    gen_loss_forward, gen_loss_params = gen_loss[0], gen_loss[1]
    disc_loss_forward, disc_loss_params = disc_loss[0], disc_loss[1]
    def forward(params, inputs, states):
        gen_params, disc_params = params
        gen_states, disc_states, gen_loss_states, disc_loss_states = states
        gen_outputs, gen_states = gen_forward(
            gen_params, inputs[1], gen_states)
        real_outputs, disc_states = disc_forward(
            disc_params, inputs[0], disc_states)
        fake_outputs, disc_states = disc_forward(
            disc_params, gen_outputs, disc_states)
        disc_outputs = [real_outputs, fake_outputs]
        net_outputs = [gen_outputs, disc_outputs]
        gen_loss_outputs, gen_loss_states = gen_loss_forward(
            gen_loss_params, fake_outputs, gen_loss_states)
        disc_loss_outputs, disc_loss_states = disc_loss_forward(
            disc_loss_params, disc_outputs, disc_loss_states)
        loss_outputs = [gen_loss_outputs, disc_loss_outputs]
        states = (gen_states, disc_states, gen_loss_states, disc_loss_states)
        return net_outputs, loss_outputs, states
    def backward(params, inputs, states):
        gen_params, disc_params = params
        gen_states, disc_states, gen_loss_states, disc_loss_states = states
        gen_loss_states, disc_loss_states = states[2], states[3]
        # Forward propagate and build backward graph
        gen_vjpf, gen_outputs, gen_states = vjp(
            gen_forward, gen_params, inputs[1], gen_states)
        disc_vjpf_real, real_outputs, disc_states = vjp(
            disc_forward, disc_params, inputs[0], disc_states)
        disc_vjpf_fake, fake_outputs, disc_states = vjp_full(
            disc_forward, disc_params, gen_outputs, disc_states)
        disc_outputs = [real_outputs, fake_outputs]
        net_outputs = [gen_outputs, disc_outputs]
        gen_loss_vjpf, gen_loss_outputs, gen_loss_states = vjp_inputs(
            gen_loss_forward, gen_loss_params, fake_outputs, gen_loss_states)
        disc_loss_vjpf, disc_loss_outputs, disc_loss_states = vjp_inputs(
            disc_loss_forward, disc_loss_params, disc_outputs, disc_loss_states)
        loss_outputs = [gen_loss_outputs, disc_loss_outputs]
        states = (gen_states, disc_states, gen_loss_states, disc_loss_states)
        # Backward propagate to generator
        grads_gen_loss_outputs = map_ones_like(gen_loss_outputs)
        grads_fake_outputs_gen = gen_loss_vjpf(grads_gen_loss_outputs)
        _, grads_gen_outputs_gen = disc_vjpf_fake(grads_fake_outputs_gen)
        grads_gen_params = gen_vjpf(grads_gen_outputs_gen)
        # Backward propagate to discriminator
        grads_disc_loss_outputs = map_ones_like(disc_loss_outputs)
        grads_disc_outputs = disc_loss_vjpf(grads_disc_loss_outputs)
        grads_real_outputs, grads_fake_outputs_disc = grads_disc_outputs
        grads_disc_params_real = disc_vjpf_real(grads_real_outputs)
        grads_disc_params_fake, _ = disc_vjpf_fake(grads_fake_outputs_disc)
        grads_disc_params = map_add(
            grads_disc_params_real, grads_disc_params_fake)
        grads = (grads_gen_params, grads_disc_params)
        return grads, net_outputs, loss_outputs, states
    return ModelTuple(forward, backward, initial_params, initial_states)


def Embed(*args):
    """Embedding model that supports segment gradients.

    Args:
      embed0, embed1, ....: a sequence of xnn.Embed modules.
      net: the network module that accepts embedded results.
      loss: the loss module.

      forward: the forward function that returns net_outputs, loss_outputs, and
        states.
      backward: the backward function that returns grads and forward returns.
        The grads is a tuple (embed_grads, net_grads) in which embed_grads can
        be used with segment optimizers, and net_grads is dense.
      initial_params: the initial parameters which is (embed_params, net_params)
      initial_states: the initial states.
    """
    if len(args) > 3:
        embed = xnn.Parallel(*args[0:-2])
    else:
        embed = args[0]
    net = args[-2]
    loss = args[-1]
    embed_forward, embed_params, embed_initial_states = embed
    net_forward, net_params, net_initial_states = net
    loss_forward, loss_params, loss_initial_states = loss
    initial_params = (embed_params, net_params)
    initial_states = (embed_initial_states, net_initial_states,
                      loss_initial_states)
    initial_states = (embed_initial_states, net_initial_states,
                      loss_initial_states)
    def forward(params, inputs, states):
        embed_params, net_params = params
        embed_inputs, net_targets = inputs
        embed_states, net_states, loss_states = states
        embed_outputs, embed_states = embed_forward(
            embed_params, embed_inputs, embed_states)
        net_outputs, net_states = net_forward(
            net_params, embed_outputs, net_states)
        loss_inputs = [net_outputs, net_targets]
        loss_outputs, loss_states = loss_forward(
            loss_params, loss_inputs, loss_states)
        states = (embed_states, net_states, loss_states)
        return [embed_outputs, net_outputs], loss_outputs, states
    def backward(params, inputs, states):
        embed_params, net_params = params
        embed_inputs, net_targets = inputs
        embed_states, net_states, loss_states = states
        # Forward propagate and build backward graph.
        embed_outputs, embed_states = embed_forward(
            embed_params, embed_inputs, embed_states)
        net_vjpf, net_outputs, net_states = vjp_full(
            net_forward, net_params, embed_outputs, net_states)
        loss_inputs = [net_outputs, net_targets]
        loss_vjpf, loss_outputs, loss_states = vjp_inputs(
            loss_forward, loss_params, loss_inputs, loss_states)
        states = (embed_states, net_states, loss_states)
        # Backward propagate.
        grads_loss_outputs = map_ones_like(loss_outputs)
        grads_net_outputs, _ = loss_vjpf(grads_loss_outputs)
        grads_net_params, grads_embed_outputs = net_vjpf(grads_net_outputs)
        # Build segment gradients for embed modules.
        if len(args) > 3:
            grads_embed_params = []
            for index, grads_value in zip(embed_inputs, grads_embed_outputs):
                grads_embed_params.append((
                    jnp.concatenate(jtree.tree_leaves(index)),
                    jnp.concatenate(jtree.tree_leaves(grads_value))))
            grads_embed_params = tuple(grads_embed_params)
        else:
            grads_embed_params = (
                jnp.concatenate(jtree.tree_leaves(embed_inputs)),
                jnp.concatenate(jtree.tree_leaves(grads_embed_outputs)))
        grads = (grads_embed_params, grads_net_params)
        return grads, [embed_outputs, net_outputs], loss_outputs, states
    return ModelTuple(forward, backward, initial_params, initial_states)


def vectorize_states(states, batch):
    # Vectorize module states individually.
    return tuple(xnn.vectorize_states(s, batch) for s in states)

def unvectorize_states(states):
    # Aggregate module states individually.
    return tuple(xnn.unvectorize_states(s) for s in states)

def vectorize(model, map_func=jax.vmap, *args, **kwargs):
    """Vectorize the model with respect to input.

    Args:
      model: the model to be vectorized.
      map_func: jax.vmap or jax.pmap.

    Returns:
      forward: vectorized forward function.
      backward: vectorized backward function. The gradients are averaged.
      params: model parameters.
      states: vectorized states.
    """
    model_forward, model_backward, initial_params, initial_states = model
    # Map over inputs and states, but not parameters.
    forward_v = map_func(model_forward, in_axes=(None, 0, 0), *args, **kwargs)
    def forward(params, inputs, states):
        batch = jtree.tree_leaves(inputs)[0].shape[0]
        states = vectorize_states(states, batch)
        net_outputs, loss_outputs, states = forward_v(params, inputs, states)
        states = unvectorize_states(states)
        return net_outputs, loss_outputs, states
    # Map over inputs and states, but not parameters.
    backward_v = map_func(model_backward, in_axes=(None, 0, 0), *args, **kwargs)
    def backward(params, inputs, states):
        batch = jtree.tree_leaves(inputs)[0].shape[0]
        states = vectorize_states(states, batch)
        grads, net_outputs, loss_outputs, states = backward_v(
            params, inputs, states)
        states = unvectorize_states(states)
        return grads, net_outputs, loss_outputs, states
    return ModelTuple(forward, backward, initial_params, initial_states)


def jit(model, *args, **kwargs):
    """Set up the model for JIT.

    Args:
      model: an xmod model.

    Returns:
      jit_model: JIT'ed model.
    """
    forward, backward, params, states = model
    return ModelTuple(jax.jit(forward, *args, **kwargs),
                      jax.jit(backward, *args, **kwargs), params, states)
