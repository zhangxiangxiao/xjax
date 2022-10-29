"""
Unit tests for xdl.
"""

from xjax import xdl

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand
from xjax import xrand
from xjax import xnn
from xjax import xmod
from xjax import xmet
from xjax import xopt

class XDLTest(absltest.TestCase):
    def setUp(self):
        # Network is a 2-layer MLP.
        def net(mode='train'):
            return xnn.Sequential(
                xnn.Linear(8, 16),
                xnn.ReLU(),
                xnn.Dropout(p=0.5, mode=mode),
                xnn.Linear(16, 4))
        # Loss is square loss.
        loss = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())
        self.train_model = xmod.Model(net('train'), loss)
        self.test_model = xmod.Model(net('test'), loss)
        # Optimizer is SGD.
        self.optimizer = xopt.SGD(
            self.train_model.params, rate=0.01, decay=0.001)
        # Metric is using an l-1 norm.
        self.metric = xmet.Metric(
            xnn.Sequential(
                xnn.Parallel(xnn.Group(1), xnn.Identity()),
                xnn.Subtract(), xnn.Abs(), xnn.Sum()))

    def test_train(self):
        # Create trainer.
        optimizer = self.optimizer
        model = self.train_model
        metric = self.metric
        # Build data.
        data = []
        for i in range(4):
            rng0, rng1 = xrand.split(2)
            data.append([jrand.normal(rng0, shape=(8,)),
                         jrand.normal(rng1, shape=(4,))])
        # Write a callback
        def callback(step, inputs, net_out, loss_out, metric_out):
            print(step, jnp.mean(loss_out[0]), jnp.mean(loss_out[1]),
                  jnp.mean(metric_out[0]), jnp.mean(metric_out[1]))
        model, optimizer, loss_outputs, metric, metric_outputs = xdl.train(
            data, model, optimizer, metric, callback)
        model, optimizer, loss_outputs, metric, metric_outputs = xdl.train(
            data, model, optimizer, metric, callback)

    def test_test(self):
        # Create tester.
        model = self.test_model
        metric = self.metric
        # Build data.
        data = []
        for i in range(4):
            rng0, rng1 = xrand.split(2)
            data.append([jrand.normal(rng0, shape=(8,)),
                         jrand.normal(rng1, shape=(4,))])
        # Write a callback
        def callback(step, inputs, net_out, loss_out, metric_out):
            print(step, jnp.mean(loss_out[0]), jnp.mean(loss_out[1]),
                  jnp.mean(metric_out[0]), jnp.mean(metric_out[1]))
        model, loss_outputs, metric, metric_outputs = xdl.test(
            data, model, metric, callback)
        model, loss_outputs, metric, metric_outputs = xdl.test(
            data, model, metric, callback)

    def test_serialize(self):
        _, _, params, _ = self.train_model
        data = xdl.dumps(params)
        loaded_params = xdl.loads(data)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     params, loaded_params)


if __name__=='__main__':
    absltest.main()
