"""
Common models defined for JAX using xjax.xnn protocol.

A model class is one that extends xjax.xnn modules with more functions such as
`value_and_grad` and `predict`. These functions are stored as function closures
in class variables, therefore different from standard class functions. In
particular, these function closures do not work with the `self` parameter.
"""

import jax
from jax import numpy as jnp


def map_ones_like(tree):
    return jax.tree_map(jnp.ones_like, tree)


def map_add(tree1, tree2):
    return jax.tree_map(jnp.add, tree1, tree2)


class Model:
    """Generic model which is simply an xjax.xnn module."""
    def __init__(self, module):
        self.forward, self.params, self.states = module

        # Delayed evaluation for properties
        self._value_and_grad = None
        self._grad = None

    def __iter__(self):
        yield self.forward
        yield self.params
        yield self.states

    @property
    def value_and_grad(self):
        if self._value_and_grad == None:
            self._value_and_grad = jax.value_and_grad(
                self.forward, has_aux=True)
        return self._value_and_grad

    @property
    def grad(self):
        if self._grad == None:
            value_and_grad = self.value_and_grad
            def _grad(*args, **kwargs):
                ((outputs, states), grads) = value_and_grad(*args, **kwargs)
                return (grads, states)
        return self._grad


class FeedForward(Model):
    """Feedforward model.

    Args:
      net: an xjax.xnn module that has learnable parameters.
      loss: an xjax.xnn module that is used as a loss. Its parameters are not
        used in either forward or grad.
    """
    def __init__(self, net, loss):
        self.net = (net[0], None, None)
        self.loss = (loss[0], loss[1], None)
        self.params = net[1]
        self.states = (net[2], loss[2])

        self._predict = None
        self._forward = None
        self._grad = None
        self._value_and_grad = None

    @property
    def predict(self):
        if self._predict == None:
            net_forward = self.net[0]
            def _predict(params, inputs, states):
                net_states, loss_states = states
                net_outputs, net_states = net_forward(
                    params, inputs, net_states)
                return net_outputs, (net_states, loss_states)
            self._predict = _predict
        return self._predict

    @property
    def forward(self):
        if self._forward == None:
            predict = self.predict
            loss_forward, loss_params = self.loss[0], self.loss[1]
            def _forward(params, inputs, states):
                net_outputs, states = predict(params, inputs, states)
                net_states, loss_states = states
                loss_outputs, loss_states = loss_forward(
                    loss_params, net_outputs, loss_states)
                return loss_outputs, (net_states, loss_states)
            self._forward = _forward
        return self._forward


class GAN(Model):
    """Generative adversarial networks model.

    Args:
      gen: the generator module.
      disc: the discriminator module.
      gen_loss: the generator loss.
      disc_loss: the discriminator loss.
    """
    def __init__(self, gen, gen_loss, disc, disc_loss):
        self.gen = (gen[0], None, None)
        self.disc = (disc[0], None, None)
        self.gen_loss = (gen_loss[0], gen_loss[1], None)
        self.disc_loss = (disc_loss[0], disc_loss[1], None)
        self.params = (gen[1], disc[1])
        self.states = (gen[2], disc[2], gen_loss[2], disc_loss[2])

        self._forward = None
        self._grad = None
        self._value_and_grad = None
        self._predict = None

    @property
    def predict(self):
        if self._predict == None:
            gen_forward, disc_forward = self.gen[0], self.disc[0]
            def _predict(params, inputs, states):
                gen_params, disc_params = params
                gen_states, disc_states = states[0], states[1]
                gen_outputs, gen_states = gen_forward(
                    gen_params, inputs, gen_states)
                real_outputs, disc_states = disc_forward(
                    disc_params, inputs, disc_states)
                fake_outputs, disc_states = disc_forward(
                    disc_Params, gen_outputs, disc_states)
                outputs = [gen_outputs, [real_outputs, fake_outputs]]
                states = (gen_states, disc_states, states[2], states[3])
                return outputs, states
            self._predict = _predict
        return self._predict

    @property
    def forward(self):
        if self._forward == None:
            predict = self.predict
            gen_loss_forward, gen_loss_params, _ = self.gen_loss
            disc_loss_forward, disc_loss_params, _ = self.disc_loss
            def _forward(params, inputs, states):
                predict_outputs, states = predict(params, inputs, states)
                real_outputs, fake_outputs = predict_outputs[1]
                gen_loss_states, disc_loss_states = states[2], states[3]
                gen_loss_outputs, gen_loss_states = gen_loss_forward(
                    gen_loss_params, fake_outputs, gen_loss_states)
                disc_loss_outputs, disc_loss_states = disc_loss_forward(
                    disc_loss_params, [_real_output, _fake_output],
                    disc_loss_states)
                outputs = [gen_loss_outputs, disc_loss_outputs]
                states = (states[0], states[1], gen_loss_states,
                          disc_loss_states)
                return outputs, states
            self._forward = _forward
        return self._forward

    @property
    def value_and_grad(self):
        if self._value_and_grad == None:
            gen_forward, disc_forward = self.gen[0], self.disc[0]
            gen_loss_forward, gen_loss_params, _ = self.genloss
            disc_loss_forward, disc_loss_params, _ = self.disc_loss
            def _value_and_grad(params, inputs, states):
                gen_params, disc_params = params
                gen_states, disc_states = states[0], states[1]
                gen_loss_states, disc_loss_states = states[2], states[3]
                # Forward propagation
                def gen_fwd(_gen_params):
                    return gen_forward(_gen_params, inputs, gen_states)
                gen_outputs, gen_vjp, gen_states = jax.vjp(
                    gen_fwd, gen_params, has_aux=True)
                def disc_fwd_real(_disc_params):
                    return disc_forward(_disc_params, inputs, disc_states)
                real_outputs, disc_vjp_real, disc_states = jax.vjp(
                    disc_fwd_real, disc_params, has_aux=True)
                def disc_fwd_fake(_disc_params, _gen_outputs):
                    return disc_forward(_disc_params, _gen_outputs, disc_states)
                fake_outputs, disc_vjp_fake, disc_states = jax.vjp(
                    disc_fwd_fake, disc_params, gen_outputs, has_aux=True)
                def gen_loss_fwd(_fake_outputs):
                    return gen_loss_forward(
                        gen_loss_params, _fake_outputs, gen_loss_states)
                gen_loss_outputs, gen_loss_vjp, gen_loss_states = jax.vjp(
                    gen_loss_fwd, fake_outputs, has_aux=True)
                def disc_loss_fwd(_real_outputs, _fake_outputs):
                    return disc_loss_forward(
                        disc_loss_params, [_real_outputs, _fake_outputs],
                        disc_loss_states)
                disc_loss_outputs, disc_loss_vjp, disc_loss_states = jax.vjp(
                    disc_loss_fwd, real_outputs, fake_outputs, has_aux=True)
                outputs = [gen_loss_outputs, disc_loss_outputs]
                states = (gen_states, disc_states, gen_loss_states,
                          disc_loss_states)
                # Backward propagation to generator
                grads_gen_loss_outputs = map_ones_like(gen_loss_outputs)
                grads_fake_outputs_gen = gen_loss_vjp(grads_gen_loss_outputs)
                _, grads_gen_outputs_gen = disc_vjp_fake(grads_fake_outputs_gen)
                grads_gen_params = gen_vjp(grads_gen_outputs_gen)
                # Backward propagation to discriminator
                grads_disc_loss_outputs = map_ones_like(disc_loss_outputs)
                grads_real_outputs, grad_fake_outputs_disc = disc_loss_vjp(
                    grads_disc_loss_outputs)
                grads_disc_params_real = disc_vjp_real(grads_real_outputs)
                grads_disc_params_fake, _ = disc_vjp_fake(
                    grads_fake_outputs_disc)
                grads_disc_params = map_add(
                    grads_disc_params_real, grads_disc_params_fake)
                return (outputs, states), (grads_gen_params, grads_disc_params)
            self._value_and_grad = _value_and_grad
        return self._value_and_grad
