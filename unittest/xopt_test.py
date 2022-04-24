"""
Unittests for xopt.
"""

from xjax import xopt

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand


class OptimizerTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        params_rng, rng = jrand.split(jrand.PRNGKey(1946))
        params_rngs = jrand.split(params_rng, 3)
        self.params = [[jrand.normal(params_rngs[0], (8,)),
                        jrand.normal(params_rngs[1], (4,))],
                       jrand.normal(params_rngs[2], (2,))]
        grads1_rng, rng = jrand.split(rng)
        grads1_rngs = jrand.split(grads1_rng, 3)
        self.grads1 = [[jrand.normal(grads1_rngs[0], (8,)),
                       jrand.normal(grads1_rngs[1], (4,))],
                      jrand.normal(grads1_rngs[2], (2,))]
        grads2_rng, rng = jrand.split(rng)
        grads2_rngs = jrand.split(grads2_rng, 3)
        self.grads2 = [[jrand.normal(grads2_rngs[0], (8,)),
                       jrand.normal(grads2_rngs[1], (4,))],
                      jrand.normal(grads2_rngs[2], (2,))]

    def test_sgd(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.SGD(params, rate=0.02, decay=0.003)
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            param = param - 0.02 * (grad1 + 0.003 * param)
            param = param - 0.02 * (grad2 + 0.003 * param)
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)


    def test_momentum(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.Momentum(
            params, rate=0.02, coeff=0.5, decay=0.003)
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            velocity = jnp.zeros_like(param)
            velocity = 0.5 * velocity + grad1 + 0.003 * param
            param = param - 0.02 * velocity
            velocity = 0.5 * velocity + grad2 + 0.003 * param
            param = param - 0.02 * velocity
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)


class VMapTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        params_rng, rng = jrand.split(jrand.PRNGKey(1946))
        params_rngs = jrand.split(params_rng, 3)
        self.params = [[jrand.normal(params_rngs[0], (8,)),
                        jrand.normal(params_rngs[1], (4,))],
                       jrand.normal(params_rngs[2], (2,))]
        grads1_rng, rng = jrand.split(rng)
        grads1_rngs = jrand.split(grads1_rng, 3)
        self.grads1 = [[jrand.normal(grads1_rngs[0], (2, 8)),
                       jrand.normal(grads1_rngs[1], (2, 4))],
                      jrand.normal(grads1_rngs[2], (2, 2))]
        grads2_rng, rng = jrand.split(rng)
        grads2_rngs = jrand.split(grads2_rng, 3)
        self.grads2 = [[jrand.normal(grads2_rngs[0], (4, 8)),
                       jrand.normal(grads2_rngs[1], (4, 4))],
                      jrand.normal(grads2_rngs[2], (4, 2))]

    def test_sgd(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.vmap(xopt.SGD(params, rate=0.02, decay=0.003))
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            param = param - 0.02 * (jnp.mean(grad1, axis=0) + 0.003 * param)
            param = param - 0.02 * (jnp.mean(grad2, axis=0) + 0.003 * param)
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)


    def test_momentum(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.vmap(xopt.Momentum(
            params, rate=0.02, coeff=0.5, decay=0.003))
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            velocity = jnp.zeros_like(param)
            velocity = 0.5 * velocity + jnp.mean(grad1, axis=0) + 0.003 * param
            param = param - 0.02 * velocity
            velocity = 0.5 * velocity + jnp.mean(grad2, axis=0) + 0.003 * param
            param = param - 0.02 * velocity
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)


if __name__ == '__main__':
    absltest.main()
