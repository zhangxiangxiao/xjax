"""
Unittests for xmod.
"""

from xjax import xmod

from absl.testing import absltest
from xjax import xnn
import jax
import jax.numpy as jnp
import jax.random as jrand


class ModuleTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, rng3, self.rng = jrand.split(jrand.PRNGKey(1946), 4)
        # net is a 2-layer MLP
        self.net = xnn.Sequential(
            xnn.Linear(rng1, 8, 4), xnn.ReLU(),
            xnn.Linear(rng2, 4, 1))
        self.model = xmod.Module(self.net)
        self.inputs = jrand.normal(rng3, shape=(8,))

    def test_forward(self):
        forward, _, params, states = self.model
        inputs = self.inputs
        net_outputs, loss_outputs, states = forward(params, inputs, states)
        self.assertIsNone(net_outputs)
        ref_forward, ref_params, ref_states = self.net
        loss_outputs_ref, ref_states = ref_forward(
            ref_params, inputs, ref_states)
        self.assertTrue(jnp.allclose(loss_outputs_ref, loss_outputs))

    def test_backward(self):
        _, backward, params, states = self.model
        inputs = self.inputs
        grads, net_outputs, loss_outputs, states = backward(
            params, inputs, states)
        loss_forward, ref_params, ref_states = self.net
        def ref_forward(_params, _inputs, _states):
            _loss_outputs, _states = loss_forward(_params, _inputs, _states)
            return jnp.sum(_loss_outputs), (_loss_outputs, _states)
        ref_backward = jax.grad(ref_forward, has_aux=True)
        ref_grads, (ref_loss_outputs, ref_states) = ref_backward(
            params, inputs, ref_states)
        self.assertTrue(jnp.allclose(ref_loss_outputs, loss_outputs))
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_grads, grads)


class ModelTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, rng3, rng4, self.rng = jrand.split(jrand.PRNGKey(1946), 5)
        # net is a 2-layer MLP.
        self.net = xnn.Sequential(
            xnn.Linear(rng1, 8, 4), xnn.ReLU(),
            xnn.Linear(rng2, 4, 2))
        # Loss is square distance.
        self.loss = xnn.Sequential(xnn.Subtract(), xnn.Norm())
        self.model = xmod.Model(self.net, self.loss)
        # inputs = [net_inputs, net_targets].
        self.inputs = [jrand.normal(rng3, shape=(8,)),
                       jrand.normal(rng4, shape=(2,))]

    def test_forward(self):
        forward, _, params, states = self.model
        inputs = self.inputs
        net_outputs, loss_outputs, states = forward(params, inputs, states)
        net_forward, net_params, net_states = self.net
        net_inputs, net_targets = inputs
        ref_net_outputs, net_states = net_forward(
            net_params, net_inputs, net_states)
        self.assertTrue(jnp.allclose(ref_net_outputs, net_outputs))
        loss_forward, loss_params, loss_states = self.loss
        loss_inputs = [net_outputs, net_targets]
        ref_loss_outputs, loss_states = loss_forward(
            loss_params, loss_inputs, loss_states)
        self.assertTrue(jnp.allclose(ref_loss_outputs, loss_outputs))
        
        
    def test_backward(self):
        forward, backward, params, states = self.model
        inputs = self.inputs
        grads, net_outputs, loss_outputs, _ = backward(
            params, inputs, states)
        ref_net_outputs, ref_loss_outputs, _ = forward(
            params, inputs, states)
        self.assertTrue(jnp.allclose(ref_net_outputs, net_outputs))
        self.assertTrue(jnp.allclose(ref_loss_outputs, loss_outputs))
        def ref_forward(_params, _inputs, _states):
            _, _loss_outputs, _ = forward(_params, _inputs, _states)
            return jnp.sum(_loss_outputs)
        ref_backward = jax.grad(ref_forward)
        ref_grads = ref_backward(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_grads, grads)


class GANTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, rng3, rng4, rng5, rng6 = jrand.split(jrand.PRNGKey(1946), 6)
        # Generator is a 2-layer MLP.
        self.gen = xnn.Sequential(
            xnn.Linear(rng2, 2, 4), xnn.ReLU(),
            xnn.Linear(rng3, 4, 8))
        # Discriminator is a linear model.
        self.disc = xnn.Linear(rng4, 8, 1)
        # Generator loss motivates the fake output to zero.
        self.gen_loss = xnn.Norm()
        # Discriminator loss motiviates the real output to zero, fake to -inf.
        self.disc_loss = xnn.Sequential(
            xnn.Parallel(xnn.Norm(), xnn.Identity()),
            xnn.Add())
        # Build the GAN model
        self.model = xmod.GAN(
            self.gen, self.disc, self.gen_loss, self.disc_loss)
        # inputs = [gen_inputs, real_inputs].
        self.inputs = [jrand.normal(rng5, shape=(2,)),
                       jrand.normal(rng6, shape=(8,))]

    def test_forward(self):
        forward, _, params, states = self.model
        inputs = self.inputs
        gen_inputs, real_inputs = inputs
        ([gen_outputs, [real_outputs, fake_outputs]],
         [gen_loss_outputs, disc_loss_outputs], states) = forward(
             params, inputs, states)
        gen_forward, gen_params, gen_states = self.gen
        ref_gen_outputs, gen_states = gen_forward(
            gen_params, gen_inputs, gen_states)
        self.assertTrue(jnp.allclose(ref_gen_outputs, gen_outputs))
        disc_forward, disc_params, disc_states = self.disc
        ref_real_outputs, disc_states = disc_forward(
            disc_params, real_inputs, disc_states)
        self.assertTrue(jnp.allclose(ref_real_outputs, real_outputs))
        ref_fake_outputs, disc_states = disc_forward(
            disc_params, gen_outputs, disc_states)
        self.assertTrue(jnp.allclose(ref_fake_outputs, fake_outputs))
        gen_loss_forward, gen_loss_params, gen_loss_states = self.gen_loss
        ref_gen_loss_outputs, gen_loss_states = gen_loss_forward(
            gen_loss_params, ref_fake_outputs, gen_loss_states)
        self.assertTrue(jnp.allclose(ref_gen_loss_outputs, gen_loss_outputs))
        disc_loss_forward, disc_loss_params, disc_loss_states = self.disc_loss
        disc_outputs = [ref_real_outputs, ref_fake_outputs]
        ref_disc_loss_outputs, disc_loss_states = disc_loss_forward(
            disc_loss_params, disc_outputs, disc_loss_states)
        self.assertTrue(jnp.allclose(ref_disc_loss_outputs, disc_loss_outputs))

    def test_backward(self):
        forward, backward, params, states = self.model
        inputs = self.inputs
        gen_inputs, real_inputs = inputs
        [gen_grads, disc_grads], net_outputs, loss_outputs, _ = backward(
            params, inputs, states)
        ref_net_outputs, ref_loss_outputs, _ = forward(
            params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_net_outputs, net_outputs)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_loss_outputs, loss_outputs)
        def ref_forward_gen(_params, _inputs, _states):
            _, [_gen_loss_outputs, _disc_loss_outputs], _ = forward(
                _params, _inputs, _states)
            return jnp.sum(_gen_loss_outputs)
        ref_backward_gen = jax.grad(ref_forward_gen)
        ref_gen_grads, _ = ref_backward_gen(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_gen_grads, gen_grads)
        def ref_forward_disc(_params, _inputs, _states):
            _, [_gen_loss_outputs, _disc_loss_outputs], _ = forward(
                _params, _inputs, _states)
            return jnp.sum(_disc_loss_outputs)
        ref_backward_disc = jax.grad(ref_forward_disc)
        _, ref_disc_grads = ref_backward_disc(params, inputs, states)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     ref_disc_grads, disc_grads)


if __name__ == '__main__':
    absltest.main()
