# XJAX
A Simple JAX framework for neural networks. Everything is done in functional programming.

XJAX is highly experimental and not ready for use, but feel free to grab and do anything to it.

## Short example
```python
from xjax import xnn, xopt, xmod, xdl, xrand
import jax.random as jrand

# A 2-layer MLP feed-forward neural net.
net = xnn.Sequential(
    xnn.Linear(8, 16),
    xnn.Dropout(p=0.5),
    xnn.ReLU(),
    xnn.Linear(16, 4))

# Square loss.
loss = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())

# Build a model using net and loss.
model = xmod.Model(net, loss)

# SGD optimizer.
optimizer = xopt.SGD(model.params, rate=0.01, decay=0.001)

# Put everything together into a learner.
train, test, states = xdl.Learner(optimizer, model)

# Create some artificial data - any iterable works.
data = []
for i in range(4):
    rng0, rng1 = xrand.split(2)
    data.append([jrand.normal(rng0, shape=(8,)),
                 jrand.normal(rng1, shape=(4,))])

# Train and test.
train_loss, _, states = train(data, states)
test_loss, _, states = test(data, states)

# Save learner states will save the model parameters and optimization states.
serialized_data = xdl.dumps(states)
```
