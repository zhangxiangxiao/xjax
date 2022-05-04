"""
Unit tests for xeval.
"""

from xjax import xeval

from absl.testing import absltest
import jax.numpy as jnp
import jax.random as jrand
from xjax import xnn
from xjax import xrand

class EvaluatorTest(absltest.TestCase):
    def setUp(self):
        # The module is a square loss
        self.module = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())
        self.evaluator = xeval.Evaluator(self.module)

    def test_evaluate(self):
        evaluate, states = self.evaluator
        forward, params, module_states = self.module
        rng1, rng2 = xrand.split(2)
        inputs = [None, jrand.normal(rng1, shape=(8,))]
        net_outputs = jrand.normal(rng2, shape=(8,))
        outputs, states = evaluate(inputs, net_outputs, states)
        module_outputs, module_states = forward(
            params, [inputs[1], net_outputs], module_states)
        self.assertTrue(jnp.allclose(module_outputs, outputs))

    def test_vmap(self):
        evaluate, states = xeval.vmap(self.evaluator, 2)
        rng1, rng2 = xrand.split(2)
        inputs = jrand.normal(rng1, shape=(2, 8))
        net_outputs = jrand.normal(rng2, shape=(2, 8))
        outputs, states = evaluate(inputs, net_outputs, states)
        self.assertEqual((2,), outputs.shape)


class BinaryEvalTest(absltest.TestCase):
    def setUp(self):
        self.evaluator = xeval.BinaryEval()

    def test_evaluate(self):
        evaluate, states = self.evaluator
        rng1, rng2, rng3 = xrand.split(3)
        inputs = [jrand.normal(rng1, shape=(8,)),
                  jnp.array([-1, 1, 1, -1])]
        net_outputs = jnp.array([-0.5, -0.1, 0.2, 1.3])
        outputs, states = evaluate(inputs, net_outputs, states)
        self.assertTrue(jnp.allclose(jnp.array(0.5), outputs))

    def test_vmap(self):
        evaluate, states = xeval.vmap(self.evaluator, 2)
        rng1, rng2, rng3 = xrand.split(3)
        inputs = [jrand.normal(rng1, shape=(2, 8)),
                  jnp.array([[-1, 1, 1, -1], [1, -1, 1, -1]])]
        net_outputs = jnp.array([[-0.5, -0.1, 0.2, 1.3],
                                 [-0.2, -0.3, -0.7, -1.1]])
        outputs, states = evaluate(inputs, net_outputs, states)
        self.assertTrue(jnp.allclose(jnp.array([0.5, 0.5]), outputs))


class ClassEvalTest(absltest.TestCase):
    def setUp(self):
        self.evaluator = xeval.ClassEval()

    def test_evaluate(self):
        evaluate, states = self.evaluator
        rng1, rng2, rng3 = xrand.split(3)
        inputs = [jrand.normal(rng1, shape=(8,)),
                  jrand.randint(rng2, shape=(), minval=0, maxval=4)]
        net_outputs = jrand.normal(rng1, shape=(4,))
        outputs, states = evaluate(inputs, net_outputs, states)
        reference = jnp.mean(jnp.equal(
            inputs[1], jnp.argmax(net_outputs, axis=-1)))
        self.assertTrue(jnp.allclose(reference, outputs))

    def test_vmap(self):
        evaluate, states = xeval.vmap(self.evaluator, 2)
        rng1, rng2, rng3 = xrand.split(3)
        inputs = [jrand.normal(rng1, shape=(2, 8)),
                  jrand.randint(rng2, shape=(2,), minval=0, maxval=4)]
        net_outputs = jrand.normal(rng1, shape=(2, 4))
        outputs, states = evaluate(inputs, net_outputs, states)
        reference = jnp.equal(
            inputs[1], jnp.argmax(net_outputs, axis=-1))
        self.assertTrue(jnp.allclose(reference, outputs))


if __name__ == '__main__':
    absltest.main()
