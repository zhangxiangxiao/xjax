"""
Functional autoregressive models library for JAX.

Design principle: an xar module is a function that return 3 objects:
  forward: the forward function.
  decode: the step-wise autoregressive decode function.
  params: the initial parameters.
  states: the initial states.

To build a module compatible with xnn, simply do
`module = (forward, params, states)`

The decode function should have the following signature
outputs, states = decode(params, inputs, candidates, states)
`inputs` is the inputs to the module, if there is any.

When using decode with xar.beam, `candidates` will be a matrix of integers with
each row containing one sequence of chosen indices from previous steps.
`outputs` should be a matrix of log-probabilities for the next search step, in
which each row is a beam candidate.

When using decode with xar.sample, `candidates` will be a vector of integers
containing the sequence of indices chosen so far. outputs should be a vector of
log-probabilities for the next step.

When using decode with xar.forward, 'candidates' will be an array of previous
outputs with the leading dimension representing the steps. `outputs` should be
an array for the next step.
"""

from __future__ import absolute_import

from collections import namedtuple

from xjax import xnn


ModuleTuple = namedtuple('Module', ['forward', 'decode', 'params', 'states'])


def Transformer():
    """Transformer autoregressive module."""
    initial_params = None
    initial_states = None
    def forward(params, inputs, states):
        pass
    def decode(params, inputs, candidates, states):
        pass
    return ModuleTuple(forward, decode, initial_params, initial_states)


def WaveNet():
    """WaveNet autoregressive module."""
    initial_params = None
    initial_states = None
    def forward(params, inputs, states):
        pass
    def decode(params, inputs, candidates, states):
        pass
    return ModuleTuple(forward, decode, initial_params, initial_states)


def beam(decode, inputs, states, max_len=16, stop_index=None):
    pass


def sample(decode, inputs, states, max_len=16, step_index=None):
    pass


def foward(decode, inputs, states, max_len=16, stop_func=None):
    pass


def get_module(module):
    """Get an xnn module from an xar module."""
    return xnn.ModuleTuple(module[0], module[2], module[3])
