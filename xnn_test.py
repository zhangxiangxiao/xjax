"""
Unittests for xnn.
"""

from xjax import xnn

from absl.testing import absltest
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import numpy as np


class LinearTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        rng, model_rng = jrand.split(rng)
        forward, params, states = xnn.Linear(model_rng, 4, 8)
        inputs = jrand.normal(rng, shape=(4,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = jnp.dot(params[0], inputs) + params[1]
        self.assertTrue(jnp.array_equal(reference, outputs))


class EmbedTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        rng, model_rng = jrand.split(rng)
        forward, params, states = xnn.Embed(model_rng, 8, 4)
        rng, inputs_rng = jrand.split(rng)
        inputs = jrand.randint(inputs_rng, (3, ), 0, 8, dtype='uint64')
        outputs, states = forward(params, inputs, states)
        self.assertEqual(2, outputs.ndim)
        self.assertEqual(3, outputs.shape[0])
        self.assertEqual(4, outputs.shape[1])
        reference = jnp.take(params, inputs, axis=0)
        self.assertTrue(jnp.array_equal(reference, outputs))


class DropoutTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        rng, model_rng = jrand.split(rng)
        forward, params, states = xnn.Dropout(model_rng)
        inputs = jrand.normal(rng, shape=(8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        rng, dropout_rng = jrand.split(model_rng)
        keep = jrand.bernoulli(dropout_rng, 0.5, inputs.shape)
        reference = jnp.where(keep, inputs / 0.5, 0)
        self.assertTrue(jnp.array_equal(reference, outputs))


class TransferTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        forward, params, states = module(*args, **kwargs)
        inputs = jrand.normal(rng, shape=(8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_abs(self):
        return self.template(xnn.Abs, jnp.abs)

    def test_tanh(self):
        return self.template(xnn.Tanh, jnp.tanh)

    def test_exp(self):
        return self.template(xnn.Exp, jnp.exp)

    def test_relu(self):
        return self.template(xnn.ReLU, jnn.relu)

    def test_sigmoid(self):
        return self.template(xnn.Sigmoid, jnn.sigmoid)

    def test_softplus(self):
        return self.template(xnn.Softplus, jnn.softplus)

    def test_log_sigmoid(self):
        return self.template(xnn.LogSigmoid, jnn.log_sigmoid)

    def test_softmax(self):
        return self.template(xnn.Softmax, jnn.softmax)

    def test_log_softmax(self):
        return self.template(xnn.LogSoftmax, jnn.log_softmax)

    def test_normalize(self):
        return self.template(xnn.Standardize, jnn.standardize)


class ReductionTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        forward, params, states = module(axis=-1, *args, **kwargs)
        inputs = jrand.normal(rng, shape=(8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs, axis=-1, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_max(self):
        return self.template(xnn.Max, jnp.max)

    def test_mean(self):
        return self.template(xnn.Mean, jnp.mean)

    def test_median(self):
        return self.template(xnn.Median, jnp.median)

    def test_min(self):
        return self.template(xnn.Min, jnp.min)

    def test_prod(self):
        return self.template(xnn.Prod, jnp.prod)

    def test_std(self):
        return self.template(xnn.Std, jnp.std)

    def test_sum(self):
        return self.template(xnn.Sum, jnp.sum)

    def test_var(self):
        return self.template(xnn.Var, jnp.var)

    def test_norm(self):
        return self.template(xnn.Norm, jnp.linalg.norm)

    def test_logsumexp(self):
        return self.template(xnn.Logsumexp, jnn.logsumexp)


class TransposeTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4, 2))
        forward, params, states = xnn.Transpose(axes=(2, 1, 0))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 4, 8), outputs.shape)
        reference = jnp.transpose(inputs, axes=(2, 1, 0))
        self.assertTrue(jnp.array_equal(reference, outputs))


class ReshapeTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4, 2))
        forward, params, states = xnn.Reshape(newshape=(-1, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((16, 4), outputs.shape)
        reference = jnp.reshape(inputs, newshape=(-1, 4))
        self.assertTrue(jnp.array_equal(reference, outputs))


class RepeatTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4))
        forward, params, states = xnn.Repeat(repeats=4, axis=-1)
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 16), outputs.shape)
        reference = jnp.repeat(inputs, repeats=4, axis=-1)
        self.assertTrue(jnp.array_equal(reference, outputs))


class IdentityTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4))
        forward, params, states = xnn.Identity()
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs, outputs))


class MulConstTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4))
        forward, params, states = xnn.MulConst(const=3.2)
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 4), outputs.shape)
        reference = inputs * 3.2
        self.assertTrue(jnp.array_equal(reference, outputs))


class AddConstTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = jrand.normal(rng, shape=(8, 4))
        forward, params, states = xnn.AddConst(const=1.87)
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 4), outputs.shape)
        reference = inputs + 1.87
        self.assertTrue(jnp.array_equal(reference, outputs))


class GroupTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rngs = jrand.split(jrand.PRNGKey(1946), 5)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        forward, params, states = xnn.Group(ind=[[0,1,2],[4,3,2]])
        outputs, states = forward(params, inputs, states)
        self.assertEqual(2, len(outputs))
        self.assertEqual(3, len(outputs[0]))
        self.assertEqual(3, len(outputs[1]))
        self.assertTrue(jnp.array_equal(inputs[0], outputs[0][0]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[0][1]))
        self.assertTrue(jnp.array_equal(inputs[2], outputs[0][2]))
        self.assertTrue(jnp.array_equal(inputs[4], outputs[1][0]))
        self.assertTrue(jnp.array_equal(inputs[3], outputs[1][1]))
        self.assertTrue(jnp.array_equal(inputs[2], outputs[1][2]))


class FlattenTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rngs = jrand.split(jrand.PRNGKey(1946), 3)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        forward, params, states = xnn.Flatten()
        outputs, states = forward(
            params, [[inputs[0], inputs[1]],[inputs[1], inputs[2]]], states)
        self.assertEqual(4, len(outputs))
        self.assertTrue(jnp.array_equal(inputs[0], outputs[0]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[1]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[2]))
        self.assertTrue(jnp.array_equal(inputs[2], outputs[3]))


class UnpackTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        inputs = [jrand.normal(rng, shape=(8,))]
        forward, params, states = xnn.Unpack()
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs[0], outputs))


class ArithmeticTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        # A Logic Named Joe
        rng1, rng2 = jrand.split(jrand.PRNGKey(1946))
        inputs1 = jrand.normal(rng1, shape=(8,))
        inputs2 = jrand.normal(rng2, shape=(8,))
        forward, params, states = module(*args, **kwargs)
        outputs, states = forward(params, [inputs1, inputs2], states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs1, inputs2, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_add(self):
        self.template(xnn.Add, jnp.add)

    def test_subtract(self):
        self.template(xnn.Subtract, jnp.subtract)

    def test_multiply(self):
        self.template(xnn.Multiply, jnp.multiply)

    def test_divide(self):
        self.template(xnn.Divide, jnp.divide)


class MatMulTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng1, rng2 = jrand.split(jrand.PRNGKey(1946))
        matrix1 = jrand.normal(rng1, shape=(8, 4))
        matrix2 = jrand.normal(rng2, shape=(4, 2))
        forward, params, states = xnn.MatMul()
        outputs, states = forward(params, [matrix1, matrix2], states)
        self.assertEqual((8, 2), outputs.shape)
        reference = jnp.matmul(matrix1, matrix2)
        self.assertTrue(jnp.array_equal(reference, outputs))


class DotTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng1, rng2 = jrand.split(jrand.PRNGKey(1946))
        matrix = jrand.normal(rng1, shape=(8, 4))
        vector = jrand.normal(rng2, shape=(4,))
        forward, params, states = xnn.Dot()
        outputs, states = forward(params, [matrix, vector], states)
        self.assertEqual((8,), outputs.shape)
        reference = jnp.dot(matrix, vector)
        self.assertTrue(jnp.array_equal(reference, outputs))


class SequentialTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng1, rng2 = jrand.split(jrand.PRNGKey(1946))
        linear = xnn.Linear(rng1, 8, 4)
        forward, params, states = xnn.Sequential(linear, xnn.ReLU(), xnn.Mean())
        inputs = jrand.normal(rng2, shape=(8,))
        outputs, states = forward(params, inputs, states)
        linear_forward, linear_params, linear_states = linear
        linear_outputs, linear_states = linear_forward(
            linear_params, inputs, linear_states)
        reference = jnp.mean(jnn.relu(linear_outputs))
        self.assertTrue(jnp.array_equal(reference, outputs))


class ParallelTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        rng1, rng2 = jrand.split(jrand.PRNGKey(1946))
        linear = xnn.Linear(rng1, 8, 4)
        forward, params, states = xnn.Parallel(linear, xnn.ReLU(), xnn.Mean())
        inputs = jrand.normal(rng2, shape=(8,))
        outputs, states = forward(params, [inputs,]*3, states)
        linear_forward, linear_params, linear_states = linear
        linear_outputs, linear_states = linear_forward(
            linear_params, inputs, linear_states)
        reference = [linear_outputs, jnn.relu(inputs), jnp.mean(inputs)]
        self.assertEqual(3, len(outputs))
        for i in range(3):
            self.assertTrue(jnp.array_equal(reference[i], outputs[i]))


class SharedParallelTest(absltest.TestCase):
    def test_forward(self):
        # A Logic Named Joe
        linear_rng, rng = jrand.split(jrand.PRNGKey(1946))
        linear = xnn.Linear(linear_rng, 8, 4)
        forward, params, states = xnn.SharedParallel(linear)
        rngs = jrand.split(rng, 3)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        outputs, states = forward(params, inputs, states)
        linear_forward, linear_params, linear_states = linear
        for i in range(3):
            reference, linear_states = linear_forward(
                linear_params, inputs[i], linear_states)
            self.assertTrue(jnp.array_equal(reference, outputs[i]))


if __name__ == '__main__':
    absltest.main()
