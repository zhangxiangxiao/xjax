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
    def setUp(self):
        # A Logic Named Joe
        self.rng, module_rng = jrand.split(jrand.PRNGKey(1946))
        self.module = xnn.Linear(module_rng, 4, 8)

    def test_forward(self):
        forward, params, states = self.module
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(4,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = jnp.dot(inputs, params[0]) + params[1]
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,8), outputs.shape)


class EmbedTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng, module_rng = jrand.split(jrand.PRNGKey(1946))
        self.module = xnn.Embed(module_rng, 8, 4)

    def test_forward(self):
        forward, params, states = self.module
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.randint(inputs_rng, (3, ), 0, 8, dtype='uint64')
        outputs, states = forward(params, inputs, states)
        self.assertEqual((3, 4), outputs.shape)
        reference = jnp.take(params, inputs, axis=0)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.randint(inputs_rng, (2, 3), 0, 8, dtype='uint64')
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 3, 4), outputs.shape)


class DropoutTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng, module_rng = jrand.split(jrand.PRNGKey(1946))
        self.module = xnn.Dropout(module_rng)

    def test_forward(self):
        forward, params, states = self.module
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        _, dropout_rng = jrand.split(self.module[2]['rng'])
        keep = jrand.bernoulli(dropout_rng, 0.5, inputs.shape)
        reference = jnp.where(keep, inputs / 0.5, 0)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2,8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 8), outputs.shape)


class TransferTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        self.module = module(*args, **kwargs)

        forward, params, states = self.module
        # A Logic Named Joe
        self.rng, inputs_rng = jrand.split(jrand.PRNGKey(1946))
        inputs = jrand.normal(inputs_rng, shape=(8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

        forward_v, params_v, states_v = xnn.vmap(self.module, 2)
        self.rng, inputs_v_rng = jrand.split(self.rng)
        inputs_v = jrand.normal(inputs_v_rng, shape=(2, 8))
        outputs_v, states_v = forward_v(params_v, inputs_v, states_v)
        self.assertEqual((2, 8), outputs_v.shape)

    def test_abs(self):
        return self.template(xnn.Abs, jnp.abs)

    def test_tanh(self):
        return self.template(xnn.Tanh, jnp.tanh)

    def test_exp(self):
        return self.template(xnn.Exp, jnp.exp)

    def test_square(self):
        return self.template(xnn.Square, jnp.square)

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
        self.module = module(axis=-1, *args, **kwargs)

        forward, params, states = self.module 
        # A Logic Named Joe
        self.rng, inputs_rng = jrand.split(jrand.PRNGKey(1946))
        inputs = jrand.normal(inputs_rng, shape=(8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs, axis=-1, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

        forward_v, params_v, states_v = xnn.vmap(self.module, 2)
        self.rng, inputs_v_rng = jrand.split(self.rng)
        inputs_v = jrand.normal(inputs_v_rng, shape=(2, 8, 4))
        outputs_v, states_v = forward(params_v, inputs_v, states_v)
        self.assertEqual((2, 8), outputs_v.shape)

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
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Transpose(axes=(2, 1, 0))

    def test_forward(self):
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4, 2))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 4, 8), outputs.shape)
        reference = jnp.transpose(inputs, axes=(2, 1, 0))
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        self.rng, inputs_rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4, 2))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 2, 4, 8), outputs.shape)


class ReshapeTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Reshape(newshape=(-1, 4))

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4, 2))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertEqual((16, 4), outputs.shape)
        reference = jnp.reshape(inputs, newshape=(-1, 4))
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4, 2))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 16, 4), outputs.shape)


class RepeatTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Repeat(repeats=4, axis=-1)

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 16), outputs.shape)
        reference = jnp.repeat(inputs, repeats=4, axis=-1)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 8, 16), outputs.shape)


class IdentityTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Identity()

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs, outputs))


class MulConstTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.MulConst(const=3.2)

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 4), outputs.shape)
        reference = inputs * 3.2
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 8, 4), outputs.shape)
        reference = inputs * 3.2
        self.assertTrue(jnp.array_equal(reference, outputs))


class AddConstTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.AddConst(const=4.7)

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8, 4))
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertEqual((8, 4), outputs.shape)
        reference = inputs + 4.7
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8, 4))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2, 8, 4), outputs.shape)
        reference = inputs + 4.7
        self.assertTrue(jnp.array_equal(reference, outputs))


class GroupTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Group(ind=[[0,1,2],[4,3,2]])

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 5)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        forward, params, states = self.module
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

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 5)
        inputs = [jrand.normal(rng, shape=(2, 8)) for rng in rngs]
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
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Flatten()
        
    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 3)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        forward, params, states = self.module
        outputs, states = forward(
            params, [[inputs[0], inputs[1]],[inputs[1], inputs[2]]], states)
        self.assertEqual(4, len(outputs))
        self.assertTrue(jnp.array_equal(inputs[0], outputs[0]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[1]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[2]))
        self.assertTrue(jnp.array_equal(inputs[2], outputs[3]))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 3)
        inputs = [jrand.normal(rng, shape=(2, 8)) for rng in rngs]
        outputs, states = forward(
            params, [[inputs[0], inputs[1]],[inputs[1], inputs[2]]], states)
        self.assertEqual(4, len(outputs))
        self.assertTrue(jnp.array_equal(inputs[0], outputs[0]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[1]))
        self.assertTrue(jnp.array_equal(inputs[1], outputs[2]))
        self.assertTrue(jnp.array_equal(inputs[2], outputs[3]))


class UnpackTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Unpack()

    def test_forward(self):
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = [jrand.normal(inputs_rng, shape=(8,))]
        forward, params, states = self.module
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs[0], outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = [jrand.normal(inputs_rng, shape=(2, 8))]
        outputs, states = forward(params, inputs, states)
        self.assertTrue(jnp.array_equal(inputs[0], outputs))


class ArithmeticTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        rng1, rng2, rng = jrand.split(rng, 3)
        inputs1 = jrand.normal(rng1, shape=(8,))
        inputs2 = jrand.normal(rng2, shape=(8,))
        forward, params, states = module(*args, **kwargs)
        outputs, states = forward(params, [inputs1, inputs2], states)
        self.assertEqual((8,), outputs.shape)
        reference = func(inputs1, inputs2, *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

        forward_v, params_v, states_v = xnn.vmap(module(*args, **kwargs), 2)
        rng1, rng2, rng = jrand.split(rng, 3)
        inputs1 = jrand.normal(rng1, shape=(2, 8))
        inputs2 = jrand.normal(rng2, shape=(2, 8))
        outputs, states = forward(params, [inputs1, inputs2], states)
        self.assertEqual((2, 8), outputs.shape)

    def test_add(self):
        self.template(xnn.Add, jnp.add)

    def test_subtract(self):
        self.template(xnn.Subtract, jnp.subtract)

    def test_multiply(self):
        self.template(xnn.Multiply, jnp.multiply)

    def test_divide(self):
        self.template(xnn.Divide, jnp.divide)


class MatMulTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.MatMul()
        
    def test_forward(self):
        rng1, rng2, self.rng = jrand.split(self.rng, 3)
        matrix1 = jrand.normal(rng1, shape=(8, 4))
        matrix2 = jrand.normal(rng2, shape=(4, 2))
        forward, params, states = self.module
        outputs, states = forward(params, [matrix1, matrix2], states)
        self.assertEqual((8, 2), outputs.shape)
        reference = jnp.matmul(matrix1, matrix2)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        rng1, rng2, self.rng = jrand.split(self.rng, 3)
        matrix1 = jrand.normal(rng1, shape=(2, 8, 4))
        matrix2 = jrand.normal(rng2, shape=(2, 4, 2))
        outputs, states = forward(params, [matrix1, matrix2], states)
        self.assertEqual((2, 8, 2), outputs.shape)


class DotTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        self.rng = jrand.PRNGKey(1946)
        self.module = xnn.Dot()

    def test_forward(self):
        rng1, rng2, self.rng = jrand.split(self.rng, 3)
        matrix = jrand.normal(rng1, shape=(8, 4))
        vector = jrand.normal(rng2, shape=(4,))
        forward, params, states = self.module
        outputs, states = forward(params, [matrix, vector], states)
        self.assertEqual((8,), outputs.shape)
        reference = jnp.dot(matrix, vector)
        self.assertTrue(jnp.array_equal(reference, outputs))

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        rng1, rng2, self.rng = jrand.split(self.rng, 3)
        matrix = jrand.normal(rng1, shape=(2, 8, 4))
        vector = jrand.normal(rng2, shape=(2, 4,))
        outputs, states = forward(params, [matrix, vector], states)
        self.assertEqual((2, 8), outputs.shape)



class RandomTest(absltest.TestCase):
    def template(self, module, func, *args, **kwargs):
        # A Logic Named Joe
        rng = jrand.PRNGKey(1946)
        module_rng, rng = jrand.split(rng)
        forward, params, states = module(
            module_rng, shape=(8, 4), *args, **kwargs)
        outputs, states = forward(params, None, states)
        self.assertEqual((8, 4), outputs.shape)
        reference_rng, _ = jrand.split(module_rng)
        reference = func(reference_rng, shape=(8, 4), *args, **kwargs)
        self.assertTrue(jnp.array_equal(reference, outputs))

        module_v_rng, rng = jrand.split(rng)
        forward_v, params_v, states_v = xnn.vmap(module(
            module_v_rng, shape=(8, 4), *args, **kwargs), 2)
        outputs, states = forward_v(params_v, None, states_v)
        self.assertEqual((2, 8, 4), outputs.shape)
        
    def test_normal(self):
        return self.template(xnn.Normal, jrand.normal)
        
    def test_uniform(self):
        return self.template(xnn.Uniform, jrand.uniform)
        
    def test_bernoulli(self):
        return self.template(xnn.Bernoulli, jrand.bernoulli)                                                                                                        


class SequentialTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, self.rng = jrand.split(jrand.PRNGKey(1946), 3)
        self.module = xnn.Sequential(
            xnn.Linear(rng1, 8, 4),
            xnn.Dropout(rng2),
            xnn.ReLU(),
            xnn.Mean())

    def test_forward(self):
        forward, params, states = self.module
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8,))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((), outputs.shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8))
        outputs, states = forward(params, inputs, states)
        self.assertEqual((2,), outputs.shape)


class ParallelTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, self.rng = jrand.split(jrand.PRNGKey(1946), 3)
        self.module = xnn.Parallel(
            xnn.Sequential(xnn.Linear(rng1, 8, 4), xnn.Dropout(rng2)),
            xnn.ReLU(),
            xnn.Mean())

    def test_forward(self):
        forward, params, states = self.module
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(8,))
        outputs, states = forward(params, [inputs,]*3, states)
        self.assertEqual(3, len(outputs))
        self.assertEqual((4,), outputs[0].shape)
        self.assertEqual((8,), outputs[1].shape)
        self.assertEqual((), outputs[2].shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        inputs = jrand.normal(inputs_rng, shape=(2, 8))
        outputs, states = forward(params, [inputs,]*3, states)
        self.assertEqual(3, len(outputs))
        self.assertEqual((2, 4), outputs[0].shape)
        self.assertEqual((2, 8), outputs[1].shape)
        self.assertEqual((2,), outputs[2].shape)


class SharedParallelTest(absltest.TestCase):
    def setUp(self):
        # A Logic Named Joe
        rng1, rng2, self.rng = jrand.split(jrand.PRNGKey(1946), 3)
        self.module = xnn.SharedParallel(xnn.Sequential(
            xnn.Linear(rng1, 8, 4), xnn.Dropout(rng2)))

    def test_forward(self):
        forward, params, states = self.module
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 3)
        inputs = [jrand.normal(rng, shape=(8,)) for rng in rngs]
        outputs, states = forward(params, inputs, states)
        self.assertEqual(3, len(outputs))
        for i in range(len(outputs)):
            self.assertEqual((4,), outputs[i].shape)

    def test_vmap(self):
        forward, params, states = xnn.vmap(self.module, 2)
        inputs_rng, self.rng = jrand.split(self.rng)
        rngs = jrand.split(inputs_rng, 3)
        inputs = [jrand.normal(rng, shape=(2, 8)) for rng in rngs]
        outputs, states = forward(params, inputs, states)
        self.assertEqual(3, len(outputs))
        for i in range(len(outputs)):
            self.assertEqual((2, 4), outputs[i].shape)


if __name__ == '__main__':
    absltest.main()
