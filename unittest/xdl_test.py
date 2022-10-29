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

class TrainerTesterTest(absltest.TestCase):
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
        self.evaluator = xmet.Metric(
            xnn.Sequential(
                xnn.Parallel(xnn.Group(1), xnn.Identity()),
                xnn.Subtract(), xnn.Abs(), xnn.Sum()))

    def test_trainer(self):
        # Create trainer.
        optimizer = self.optimizer
        model = self.train_model
        evaluator = self.evaluator
        train, states = xdl.Trainer(optimizer, model, evaluator)
        # Build data.
        data = []
        for i in range(4):
            rng0, rng1 = xrand.split(2)
            data.append([jrand.normal(rng0, shape=(8,)),
                         jrand.normal(rng1, shape=(4,))])
        # Write a callback
        def callback(step, states, inputs, net_out, loss_out, eval_out):
            print(step, jnp.mean(loss_out[0]), jnp.mean(loss_out[1]),
                  jnp.mean(eval_out[0]), jnp.mean(eval_out[1]))
        loss_outputs, eval_outputs, states = train(data, states, callback)
        loss_outputs, eval_outputs, states = train(data, states, callback)

    def test_tester(self):
        # Create tester.
        model = self.test_model
        evaluator = self.evaluator
        test, states = xdl.Tester(model, evaluator)
        # Build data.
        data = []
        for i in range(4):
            rng0, rng1 = xrand.split(2)
            data.append([jrand.normal(rng0, shape=(8,)),
                         jrand.normal(rng1, shape=(4,))])
        # Write a callback
        def callback(step, states, inputs, net_out, loss_out, eval_out):
            print(step, jnp.mean(loss_out[0]), jnp.mean(loss_out[1]),
                  jnp.mean(eval_out[0]), jnp.mean(eval_out[1]))
        loss_outputs, eval_outputs, states = test(data, states, callback)
        loss_outputs, eval_outputs, states = test(data, states, callback)

    def test_serialize(self):
        # Create trainer.
        optimizer = self.optimizer
        model = self.train_model
        evaluator = self.evaluator
        _, states = xdl.Trainer(optimizer, model, evaluator)
        data = xdl.dumps(states)
        loaded_states = xdl.loads(data)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     states, loaded_states)


if __name__=='__main__':
    absltest.main()
