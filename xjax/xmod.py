"""
Common models defined for JAX using xjax.xnn protocol.

A model provides 'backward' function in addition to `forward`, `params` and
 `states` as in xjax.xnn. A model function as the following signature:
forward, backward, params, states = model

The `backward` function has the following signature:
`grads, outputs, new_states = backward(params, inputs, states)`
Parameter gradients are stored in `grads`.
"""

import jax
from jax import numpy as jnp


def vjp(forward, params, inputs, states):
    """jax.vjp against the params of forward."""
    def _forward(_params):
        return forward(_params, inputs, states)
    outputs, vjpf, states = jax.vjp(_forward, params, has_aux=True)
    return vjpf, outputs, states


def vjp_full(forward, params, inputs, states):
    """jax.vjp against params and inputs of forward."""
    def _forward(_params, _inputs):
        return forward(_params, _inputs, states)
    outputs, vjp, states = jax.vjp(_forward, params, inputs, has_aux=True)
    return vjpf, outputs, states


def vjp_inputs(forward, params, inputs, states):
    """jax.vjp against the inputs of forward."""
    def _forward(_inputs):
        return forward(params, _inputs, states)
    outputs, vjpf, states = jax.vjp(_forward, inputs, has_aux=True)
    return vjpf, outputs, states


def map_ones_like(tree):
    return jax.tree_map(jnp.ones_like, tree)


def map_add(tree1, tree2):
    return jax.tree_map(jnp.add, tree1, tree2)


def Model(module):
    """Generic model which is simply an xjax.xnn module."""
    forward, initial_params, initial_states = module[0]
    def backward(params, inputs, states):
        vjpf, outputs, states = vjp(forward, params, inputs, states)
        grads_outputs = map_ones_like(outputs)
        grads = vjpf(grads_outputs)
        return grads, outputs, states
    return forward, backward, initial_params, initial_states


def FeedForward(net, loss):
    """Feedforward model.

    Args:
      net: an xjax.xnn module that has learnable parameters.
      loss: an xjax.xnn module that is used as a loss.

    Returns:
      forward: the forward function that returns outputs and states. The outputs
        is a tuple of net outputs and loss outputs.
      backward: the backward function.
      params: the initial parameters from net.

    """
    net_forward, initial_params, net_initial_states = net
    loss_forward, loss_params, loss_initial_states = loss
    initial_states = (net_initial_states, loss_initial_states)
    def forward(params, inputs, states):
        net_states, loss_states = states
        net_outputs, net_states = net_forward(params, inputs, net_states)
        loss_outputs, loss_states = loss_forward(
            loss_params, net_outputs, loss_states)
        return (net_outputs, loss_outputs), (net_states, loss_states)
    def backward(params, inputs, states):
        # Forward propagate and build backward graph.
        net_states, loss_states = states
        net_vjpf, net_outputs, net_states = vjp(
            net_forward, params, inputs, net_states)
        loss_vjpf, loss_outputs, loss_states = vjp_inputs(
            loss_forward, loss_params, net_outputs, loss_states)
        outputs = (net_outputs, loss_outputs)
        states = (net_states, loss_states)
        # Backward propagate through loss.
        grads_loss_outputs = map_ones_like(loss_outputs)
        grads_net_outputs = loss_vjpf(grads_loss_outputs)
        grads = net_vjpf(grads_net_outputs)
        return grads, outputs, states
    return forward, backward, initial_params, initial_states


class GAN(Model):
    """Generative adversarial networks model.

    Args:
      gen: the generator module.
      disc: the discriminator module.
      gen_loss: the generator loss.
      disc_loss: the discriminator loss.
    """
    def __init__(self, gen, disc, gen_loss, disc_loss):
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
                    disc_params, gen_outputs, disc_states)
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
                    disc_loss_params, [real_outputs, fake_outputs],
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
                # Forward propagate and build backward graph
                gen_outputs, gen_backward, gen_states = backward_params(
                    gen_forward, gen_params, inputs, gen_states)
                real_outputs, disc_backward_real, disc_states = backward_params(
                    disc_forward, disc_params, inputs, disc_states)
                fake_outputs, disc_backward_fake, disc_states = backward(
                    disc_forward, disc_params, gen_outputs, disc_states)
                gen_loss_outputs, gen_loss_backward, gen_loss_states = (
                    backward_inputs(gen_loss_forward, gen_loss_params,
                                    fake_outputs, gen_loss_states))
                disc_loss_outputs, disc_loss_backward, disc_loss_states = (
                    backward_inputs(disc_loss_forward, disc_loss_params,
                                    [real_outputs, fake_outputs],
                                    disc_loss_states))
                outputs = [gen_loss_outputs, disc_loss_outputs]
                states = (gen_states, disc_states, gen_loss_states,
                          disc_loss_states)
                # Backward propagate to generator
                grads_gen_loss_outputs = map_ones_like(gen_loss_outputs)
                grads_fake_outputs_gen = gen_loss_backward(
                    grads_gen_loss_outputs)
                _, grads_gen_outputs_gen = disc_backward_fake(
                    grads_fake_outputs_gen)
                grads_gen_params = gen_backward(grads_gen_outputs_gen)
                # Backward propagate to discriminator
                grads_disc_loss_outputs = map_ones_like(disc_loss_outputs)
                grads_real_outputs, grad_fake_outputs_disc = disc_loss_backward(
                    grads_disc_loss_outputs)
                grads_disc_params_real = disc_backward_real(grads_real_outputs)
                grads_disc_params_fake, _ = disc_backward_fake(
                    grads_fake_outputs_disc)
                grads_disc_params = map_add(
                    grads_disc_params_real, grads_disc_params_fake)
                return (outputs, states), (grads_gen_params, grads_disc_params)
            self._value_and_grad = _value_and_grad
        return self._value_and_grad

class ATNNFAE(Model):
    """Adversarially-Trained Normalized Noisy-Feature Auto-Encoder

    Args:
      enc: the encoder module.
      dec: the decoder / generator module.
      disc: the discriminator module.
      inj: the noise injection modle.
      rnd: the random noise generation module.
      ae_loss: the autoencode loss.
      gen_loss: the generator loss.
      disc_loss: the discriminator loss.
    """
    def __init__(self, enc, dec, disc, inj, rnd, ae_loss, gen_loss, disc_loss):
        self.enc = (enc[0], None, None)
        self.dec = (dec[0], None, None)
        self.disc = (disc[0], None, None)
        self.inj = (inj[0], inj[1], None)
        self.rnd = (rnd[0], rnd[1], None)
        self.ae_loss = (ae_loss[0], ae_loss[1], None)
        self.gen_loss = (gen_loss[0], gen_loss[1], None)
        self.disc_loss = (disc_loss[0], disc_loss[1], None)
        self.params = (enc[1], dec[1], disc[1])
        self.states = (enc[2], dec[2], disc[2], inj[2], rnd[2], ae_loss[2],
                       gen_loss[2], disc_loss[2])

        self._forward = None
        self._grad = None
        self._value_and_grad = None
        self._predict = None


    @property
    def predict(self):
        if self._predict == None:
            enc_forward, dec_forward = self.enc[0], self.dec[0]
            inj_forward, rnd_forward = self.inj[0], self.rnd[0]
            disc_forward = self.disc[0]
            def _predict(params, inputs, states):
                enc_params, dec_params, disc_params = params
                enc_states, dec_states, disc_states = states[:3]
                inj_states, rnd_states = states[3:5]
                enc_outputs, enc_states = enc_forward(
                    enc_params, inputs, enc_states)
                inj_outputs, inj_states = inj_forward(
                    inj_params, enc_outputs, inj_states)
                rnd_outputs, rnd_states = rnd_forward(
                    rnd_params, enc_outputs, rnd_states)
                dec_outputs, dec_states = dec_forward(
                    dec_params, inj_outputs, dec_states)
                gen_outputs, dec_states = dec_forward(
                    gen_params, rnd_outputs, dec_states)
                real_outputs, disc_states = disc_forward(
                    disc_params, dec_outputs, disc_states)
                fake_outputs, disc_states = disc_forward(
                    disc_Params, gen_outputs, disc_states)
                outputs = [[dec_outputs, gen_outputs],
                           [real_outputs, fake_outputs]]
                states = (enc_states, dec_states, disc_states, inj_states,
                          rnd_states) + states[5:]
                return outputs, states
            self._predict = _predict
        return self._predict

    @property
    def forward(self):
        if self._forward == None:
            predict = self.predict
            ae_loss_forward, ae_loss_params, _ = self.ae_loss
            gen_loss_forward, gen_loss_params, _ = self.gen_loss
            disc_loss_forward, disc_loss_params, _ = self.disc_loss
            def _forward(params, inputs, states):
                predict_outputs, states = predict(params, inputs, states)
                [[dec_outputs, gen_outputs], [real_outputs, fake_outputs]] = (
                    predict_outputs)
                ae_loss_states, gen_loss_states, disc_loss_states = states[5:]
                ae_loss_outputs, ae_loss_states = ae_loss_forward(
                    ae_loss_params, dec_outputs, ae_loss_states)
                gen_loss_outputs, gen_loss_states = gen_loss_forward(
                    gen_loss_params, fake_outputs, gen_loss_states)
                disc_loss_outputs, disc_loss_states = disc_loss_forward(
                    disc_loss_params, [real_outputs, fake_outputs],
                    disc_loss_states)
                outputs = [ae_loss_outputs, gen_loss_outputs, disc_loss_outputs]
                states = states[0:5] + (ae_loss_states, gen_loss_states,
                                        disc_loss_states)
                return outputs, states
            self._forward = _forward
        return self._forward

    @property
    def value_and_grad(self):
        if self._value_and_grad == None:
            enc_forward, dec_forward = self.enc[0], self.dec[0]
            inj_forward, rnd_forward = self.inj[0], self.rnd[0]
            ae_loss_forward, ae_loss_params, _ = self.ae_loss
            gen_loss_forward, gen_loss_params, _ = self.gen_loss
            disc_loss_forward, disc_loss_params, _ = self.disc_loss
            def _value_and_grad(params, inputs, states):
                enc_params, dec_params, disc_params = params
                enc_states, dec_states, disc_states = states[:3]
                inj_states, rnd_states = states[3:5]
                ae_loss_states, gen_loss_states, disc_loss_states = states[5:]
                # Forward propagate and build backward graph.
                enc_outputs, enc_backward, ens_states = backward_params(
                    enc_forward, enc_params, inputs, enc_states)
                inj_outputs, inj_backward, inj_states = backward_inputs(
                    inj_forward, inj_params, enc_outputs, inj_states)
                rnd_outputs, rnd_states = rnd_forward(
                    rnd_params, enc_outputs, rnd_states)
                dec_outputs, dec_backward, dec_states = backward(
                    dec_forward, dec_params, inj_outputs, dec_states)
                gen_outputs, gen_backward, dec_states = backward_params(
                    dec_forward, dec_params, rnd_outputs, dec_states)
                real_outputs, disc_backward_real, disc_states = backward_params(
                    disc_forward, disc_params, dec_outputs, disc_states)
                fake_outputs, disc_backward_fake,  disc_states = backward(
                    disc_forward, disc_params, gen_outputs, disc_states)
                ae_loss_outputs, ae_loss_backward, ae_loss_states = (
                    backward_inputs(ae_loss_forward, ae_loss_params,
                                    dec_outputs, ae_loss_states))
                gen_loss_outputs, gen_loss_backward, gen_loss_states = (
                    backward_inputs(gen_loss_forward, gen_loss_params,
                                    fake_outputs, gen_loss_states))
                disc_loss_outputs, disc_loss_backward, disc_loss_states = (
                    backward_inputs(disc_loss_forward, disc_loss_params,
                                    [real_outputs, fake_outputs],
                                    disc_loss_states))
                outputs = [ae_loss_outputs, gen_loss_outputs, disc_loss_outputs]
                states = [enc_states, dec_states, disc_states, inj_states,
                          rnd_states, ae_loss_states, gen_loss_states,
                          disc_loss_states]
                # Backward propagate to autoencoder.
                grads_ae_loss_outputs = map_ones_like(ae_loss_outputs)
                grads_dec_outputs = ae_loss_backward(grads_ae_loss_outputs)
                grads_dec_params_ae, grads_inj_outputs = dec_backward(
                    grads_dec_outputs)
                grads_enc_outputs = inj_backward(grads_inj_outputs)
                grads_enc_params = enc_backward(grads_enc_outputs)
                # Backward propagate to generator.
                grads_gen_loss_outputs = map_ones_like(gen_loss_outputs)
                grads_fake_outputs_gen = gen_loss_backward(
                    grads_gen_loss_outputs)
                _, grads_gen_outputs = disc_backward_fake(
                    grads_fake_outputs_gen)
                grads_dec_params_gen = gen_backward(grads_gen_outputs)
                # Backward propagate to discriminator
                grads_disc_loss_outputs = map_ones_like(disc_loss_outputs)
                grads_real_outputs, grads_fake_outputs_disc = (
                    disc_loss_backward(grads_disk_loss_outputs))
                grads_disc_params_real = disc_backward_real(grads_real_outputs)
                grads_disc_params_fake, _ = disc_backward_fake(
                    grads_fake_outputs_disc)
                # Add parameters together
                grads_dec_params = map_add(
                    grads_dec_params_ae, grads_dec_params_gen)
                grads_disc_params = map_add(
                    grads_disc_params_real, grads_disc_params_fake)
                grads = (grads_enc_params, grads_dec_params, grads_disc_params)
                return (outputs, states), grads
            self._value_and_grad = _value_and_grad
        return self._value_and_grad
