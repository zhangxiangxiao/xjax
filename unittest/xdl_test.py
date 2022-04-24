"""
Unit tests for xdl.
"""

from xjax import xdl

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.random as jrand
from xjax import xnn
from xjax import xmod
from xjax import xeval
from xjax import xopt

class LearnerTest(absltest.TestCase):
    def setUp(self):
        # Logic Named Joe
        rng0, rng1, rng2, self.rng = jrand.split(jrand.PRNGKey(1946), 4)
        # Network is a 2-layer MLP.
        def net(mode='train'):
            return xnn.Sequential(
                xnn.Linear(rng0, 8, 16),
                xnn.ReLU(),
                xnn.Dropout(rng1, p=0.5, mode=mode),
                xnn.Linear(rng2, 16, 4))
        # Loss is square loss.
        loss = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())
        self.train_model = xmod.Model(net('train'), loss)
        self.test_model = xmod.Model(net('test'), loss)
        # Optimizer is SGD.
        self.optimizer = xopt.SGD(
            self.train_model.params, rate=0.01, decay=0.001)
        # Evaluator is using an l-1 norm.
        self.evaluator = xeval.Evaluator(
            xnn.Sequential(xnn.Subtract(), xnn.Abs(), xnn.Sum()))

    def test_train(self):
        # Create learner.
        optimizer = self.optimizer
        train_model = self.train_model
        test_model = self.test_model
        evaluator = self.evaluator
        train, _, states = xdl.Learner(
            optimizer, train_model, test_model, evaluator)
        # Build data.
        data = []
        for i in range(4):
            rng, self.rng = jrand.split(self.rng, 2)
            data.append([jrand.normal(rng, shape=(8,)),
                         jrand.normal(rng, shape=(4,))])
        # Write a callback
        def callback(step, params, grads, net_out, loss_out, eval_out,
                     total_loss, total_eval):
            print(step, jnp.mean(loss_out), jnp.mean(eval_out),
                  jnp.mean(total_loss), jnp.mean(total_eval))
        total_loss_outputs, total_eval_outputs, states = train(
            data, states, callback)
        total_loss_outputs, total_eval_outputs, states = train(
            data, states, callback)

    def test_test(self):
        # Create learner.
        optimizer = self.optimizer
        train_model = self.train_model
        test_model = self.test_model
        evaluator = self.evaluator
        _, test, states = xdl.Learner(
            optimizer, train_model, test_model, evaluator)
        # Build data.
        data = []
        for i in range(4):
            rng, self.rng = jrand.split(self.rng, 2)
            data.append([jrand.normal(rng, shape=(8,)),
                         jrand.normal(rng, shape=(4,))])
        # Write a callback
        def callback(step, net_out, loss_out, eval_out, total_loss, total_eval):
            print(step, jnp.mean(loss_out), jnp.mean(eval_out),
                  jnp.mean(total_loss), jnp.mean(total_eval))
        total_loss_outputs, total_eval_outputs, states = test(
            data, states, callback)
        total_loss_outputs, total_eval_outputs, states = test(
            data, states, callback)

    def test_serialize(self):
        # Create learner.
        optimizer = self.optimizer
        train_model = self.train_model
        test_model = self.test_model
        evaluator = self.evaluator
        _, _, states = xdl.Learner(
            optimizer, train_model, test_model, evaluator)
        data = xdl.dumps(states)
        loaded_states = xdl.loads(data)
        jax.tree_map(lambda x, y: self.assertTrue(jnp.allclose(x, y)),
                     states, loaded_states)


if __name__=='__main__':
    absltest.main()
