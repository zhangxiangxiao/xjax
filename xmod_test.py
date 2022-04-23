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
        rng1, rng2, rng3, rng4, self.rng = jrand.split(jrand.PRNGKey(1946), 5)
        # Generator is a 2-layer MLP.
        self.gen = xnn.Sequential(
            xnn.Normal(rng1, shape=(2,)),
            xnn.Linear(rng2, 2, 4), xnn.ReLU(),
            xnn.Linear(rng3, 4, 8))
        # Discriminator is a linear model.
        self.disc = xnn.Linear(rng3, 8, 1)
        # Generator loss motivates the fake output to zero.
        self.gen_loss = xnn.Norm()
        # Discriminator loss motiviates the real output to zero, fake to -inf.
        self.disc_loss = xnn.Parallel(xnn.Norm(), xnn.Identity)
        # Build the GAN model
        self.model = xmod.GAN(self.net, self.loss)
        # inputs = [net_inputs, net_targets].
        self.inputs = [jrand.normal(rng4, shape=(8,))]
    def test_forward(self):
        pass
    def test_backward(self):
        pass


if __name__ == '__main__':
    absltest.main()
