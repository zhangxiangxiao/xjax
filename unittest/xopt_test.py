"""
Unittests for xopt.
"""

from xjax import xopt

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand
from xjax import xrand

class OptimizerTest(absltest.TestCase):
    def setUp(self):
        self.params = [[jrand.normal(xrand.split(), (8,)),
                        jrand.normal(xrand.split(), (4,))],
                       jrand.normal(xrand.split(), (2,))]
        self.grads1 = [[jrand.normal(xrand.split(), (8,)),
                       jrand.normal(xrand.split(), (4,))],
                      jrand.normal(xrand.split(), (2,))]
        self.grads2 = [[jrand.normal(xrand.split(), (8,)),
                       jrand.normal(xrand.split(), (4,))],
                      jrand.normal(xrand.split(), (2,))]

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

    def test_container(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        params0, params1 = params[0], params[1]
        update, states = xopt.Container(
            xopt.SGD(params[0], rate=0.02, decay=0.003),
            xopt.SGD(params[1], rate=0.01, decay=0.001))
        params, states = update(params, grads1, states)
        params, states = update(params, grads2, states)
        update0, states0 = xopt.SGD(params0, rate=0.02, decay=0.003)
        params0, states0 = update0(params0, grads1[0], states0)
        params0, states0 = update0(params0, grads2[0], states0)
        update1, states1 = xopt.SGD(params1, rate=0.01, decay=0.001)
        params1, states1 = update1(params1, grads1[1], states1)
        params1, states1 = update1(params1, grads2[1], states1)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     [params0, params1], params)


class SegmentTest(absltest.TestCase):
    def setUp(self):
        self.params = [[jrand.normal(xrand.split(), (64, 8)),
                        jrand.normal(xrand.split(), (64, 4))],
                       jrand.normal(xrand.split(), (64, 2))]
        self.grads1 = [[(jrand.randint(xrand.split(), (2,), 0, 64),
                         jrand.normal(xrand.split(), (2, 8))),
                       (jrand.randint(xrand.split(), (2,), 0, 64),
                        jrand.normal(xrand.split(), (2, 4)))],
                      (jrand.randint(xrand.split(), (2,), 0, 64),
                       jrand.normal(xrand.split(), (2, 2)))]
        self.grads2 = [[(jrand.randint(xrand.split(), (2,), 0, 64),
                         jrand.normal(xrand.split(), (2, 8))),
                       (jrand.randint(xrand.split(), (2,), 0, 64),
                        jrand.normal(xrand.split(), (2, 4)))],
                      (jrand.randint(xrand.split(), (2,), 0, 64),
                       jrand.normal(xrand.split(), (2, 2)))]

    def test_sgd(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.SGD(params, rate=0.02, decay=0.003)
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            index, grad_value = grad1
            param_value = jnp.take(param, index, axis=0)
            param = param.at[index].add(
                -0.02 * (grad_value + 0.003 * param_value))
            index, grad_value = grad2
            param_value = jnp.take(param, index, axis=0)
            param = param.at[index].add(
                -0.02 * (grad_value + 0.003 * param_value))
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
            index, grad_value = grad1
            param_value = jnp.take(param, index, axis=0)
            velocity_value = jnp.take(velocity, index, axis=0)
            velocity_value = (0.5 * velocity_value + grad_value +
                              0.003 * param_value)
            velocity = velocity.at[index].set(velocity_value)
            param = param.at[index].add(-0.02 * velocity_value)
            index, grad_value = grad2
            param_value = jnp.take(param, index, axis=0)
            velocity_value = jnp.take(velocity, index, axis=0)
            velocity_value = (0.5 * velocity_value + grad_value +
                              0.003 * param_value)
            velocity = velocity.at[index].set(velocity_value)
            param = param.at[index].add(-0.02 * velocity_value)
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)


    def test_container(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        params0, params1 = params[0], params[1]
        update, states = xopt.Container(
            xopt.SGD(params[0], rate=0.02, decay=0.003),
            xopt.SGD(params[1], rate=0.01, decay=0.001))
        params, states = update(params, grads1, states)
        params, states = update(params, grads2, states)
        update0, states0 = xopt.SGD(params0, rate=0.02, decay=0.003)
        params0, states0 = update0(params0, grads1[0], states0)
        params0, states0 = update0(params0, grads2[0], states0)
        update1, states1 = xopt.SGD(params1, rate=0.01, decay=0.001)
        params1, states1 = update1(params1, grads1[1], states1)
        params1, states1 = update1(params1, grads2[1], states1)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     [params0, params1], params)



class VectorizeOptimizerTest(absltest.TestCase):
    def setUp(self):
        self.params = [[jrand.normal(xrand.split(), (8,)),
                        jrand.normal(xrand.split(), (4,))],
                       jrand.normal(xrand.split(), (2,))]
        self.grads1 = [[jrand.normal(xrand.split(), (2, 8)),
                       jrand.normal(xrand.split(), (2, 4))],
                      jrand.normal(xrand.split(), (2, 2))]
        self.grads2 = [[jrand.normal(xrand.split(), (4, 8)),
                       jrand.normal(xrand.split(), (4, 4))],
                      jrand.normal(xrand.split(), (4, 2))]

    def test_sgd(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.vectorize(xopt.SGD(
            params, rate=0.02, decay=0.003))
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
        update, states = xopt.vectorize(xopt.Momentum(
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

    def test_container(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        params0, params1 = params[0], params[1]
        update, states = xopt.vectorize(xopt.Container(
            xopt.SGD(params[0], rate=0.02, decay=0.003),
            xopt.SGD(params[1], rate=0.01, decay=0.001)))
        params, states = update(params, grads1, states)
        params, states = update(params, grads2, states)
        update0, states0 = xopt.vectorize(xopt.SGD(
            params0, rate=0.02, decay=0.003))
        params0, states0 = update0(params0, grads1[0], states0)
        params0, states0 = update0(params0, grads2[0], states0)
        update1, states1 = xopt.vectorize(xopt.SGD(
            params1, rate=0.01, decay=0.001))
        params1, states1 = update1(params1, grads1[1], states1)
        params1, states1 = update1(params1, grads2[1], states1)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     [params0, params1], params)



class VectorizeSegmentTest(absltest.TestCase):
    def setUp(self):
        self.params = [[jrand.normal(xrand.split(), (64, 8)),
                        jrand.normal(xrand.split(), (64, 4))],
                       jrand.normal(xrand.split(), (64, 2))]
        self.grads1 = [[(jrand.randint(xrand.split(), (2, 2), 0, 64),
                         jrand.normal(xrand.split(), (2, 2, 8))),
                       (jrand.randint(xrand.split(), (2, 2), 0, 64),
                        jrand.normal(xrand.split(), (2, 2, 4)))],
                      (jrand.randint(xrand.split(), (2, 2), 0, 64),
                       jrand.normal(xrand.split(), (2, 2, 2)))]
        self.grads2 = [[(jrand.randint(xrand.split(), (2, 2), 0, 64),
                         jrand.normal(xrand.split(), (2, 2, 8))),
                       (jrand.randint(xrand.split(), (2, 2), 0, 64),
                        jrand.normal(xrand.split(), (2, 2, 4)))],
                      (jrand.randint(xrand.split(), (2, 2), 0, 64),
                       jrand.normal(xrand.split(), (2, 2, 2)))]

    def test_sgd(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.vectorize(xopt.SGD(
            params, rate=0.02, decay=0.003))
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            index, grad_value = grad1
            index = jnp.reshape(index, (-1,))
            grad_value = jnp.reshape(grad_value, (-1,) + param.shape[1:]) / 2
            param_value = jnp.take(param, index, axis=0)
            param = param.at[index].add(
                -0.02 * (grad_value + 0.003 * param_value))
            index, grad_value = grad2
            index = jnp.reshape(index, (-1,))
            grad_value = jnp.reshape(grad_value, (-1,) + param.shape[1:]) / 2
            param_value = jnp.take(param, index, axis=0)
            param = param.at[index].add(
                -0.02 * (grad_value + 0.003 * param_value))
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)

    def test_momentum(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        update, states = xopt.vectorize(xopt.Momentum(
            params, rate=0.02, coeff=0.5, decay=0.003))
        self.assertEqual(0, states[0])
        params, states = update(params, grads1, states)
        self.assertEqual(1, states[0])
        params, states = update(params, grads2, states)
        self.assertEqual(2, states[0])
        def test_result(param, grad1, grad2, result):
            velocity = jnp.zeros_like(param)
            index, grad_value = grad1
            index = jnp.reshape(index, (-1,))
            grad_value = jnp.reshape(grad_value, (-1,) + param.shape[1:]) / 2
            param_value = jnp.take(param, index, axis=0)
            velocity_value = jnp.take(velocity, index, axis=0)
            velocity_value = (0.5 * velocity_value + grad_value +
                              0.003 * param_value)
            velocity = velocity.at[index].set(velocity_value)
            param = param.at[index].add(-0.02 * velocity_value)
            index, grad_value = grad2
            index = jnp.reshape(index, (-1,))
            grad_value = jnp.reshape(grad_value, (-1,) + param.shape[1:]) / 2
            param_value = jnp.take(param, index, axis=0)
            velocity_value = jnp.take(velocity, index, axis=0)
            velocity_value = (0.5 * velocity_value + grad_value +
                              0.003 * param_value)
            velocity = velocity.at[index].set(velocity_value)
            param = param.at[index].add(-0.02 * velocity_value)
            self.assertTrue(jnp.allclose(param, result))
        jax.tree_map(test_result, self.params, self.grads1, self.grads2, params)

    def test_container(self):
        params, grads1, grads2 = self.params, self.grads1, self.grads2
        params0, params1 = params[0], params[1]
        update, states = xopt.vectorize(xopt.Container(
            xopt.SGD(params[0], rate=0.02, decay=0.003),
            xopt.SGD(params[1], rate=0.01, decay=0.001)))
        params, states = update(params, grads1, states)
        params, states = update(params, grads2, states)
        update0, states0 = xopt.vectorize(xopt.SGD(
            params0, rate=0.02, decay=0.003))
        params0, states0 = update0(params0, grads1[0], states0)
        params0, states0 = update0(params0, grads2[0], states0)
        update1, states1 = xopt.vectorize(xopt.SGD(
            params1, rate=0.01, decay=0.001))
        params1, states1 = update1(params1, grads1[1], states1)
        params1, states1 = update1(params1, grads2[1], states1)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     [params0, params1], params)


class ScheduleTest(absltest.TestCase):
    def test_inverse_time_schedule(self):
        schedule = xopt.InverseTimeSchedule(0.1, 0.001, 0.01)
        rate = schedule(67)
        ratio = 1 / (1 + 0.01 * 67)
        ref_rate = ratio * 0.1 + (1 - ratio) * 0.001
        self.assertAlmostEqual(ref_rate, rate)


if __name__ == '__main__':
    absltest.main()
