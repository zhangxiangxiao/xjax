# XJAX
A Simple JAX framework for neural networks. Everything is done in functional programming.

XJAX is highly experimental and not ready for use, but feel free to grab and do anything to it.

## Short example
```python
from xjax import xnn, xopt, xmod, xdl, xrand
import jax.random as jrand

# Data with 6 samples - input is an 8-dim vector, target 4-dim.
data = []
for i in range(6):
    rng0, rng1 = xrand.split(2)
    data.append([jrand.normal(rng0, shape=(8,)),
                 jrand.normal(rng1, shape=(4,))])

# Model is a 2-layer MLP feed-forward neural net with square loss.
net = xnn.Sequential(
    xnn.Linear(8, 16),
    xnn.Dropout(p=0.5),
    xnn.ReLU(),
    xnn.Linear(16, 4))
loss = xnn.Sequential(xnn.Subtract(), xnn.Square(), xnn.Sum())
model = xmod.Model(net, loss)

# Train and test using SGD optimizer.
optimizer = xopt.SGD(model.params, rate=0.01, decay=0.001)
model, optimizer, train_loss, _, _ = xdl.train(data, model, optimizer)
model, test_loss, _, _ = xdl.test(data, model)

# Save the model parameters and optimization states.
serialized_data = xdl.dumps({
    'params': model.params,
    'model_states': model.states,
    'optimizer_states': optimizer.states})
```
