from flax.linen import Module, Dense, Sequential, relu, softmax
from jax.random import PRNGKey
import jax.numpy as numpy
from jax import Array
import optax

import awkward
from numpy import genfromtxt
from numpy.lib.recfunctions import structured_to_unstructured
import numpy as onp

from deepset import deepset


def id(xs):
  return xs


def MLP(features, activations):
  laypairs = \
    [ [ Dense(feats) , act ] \
      for feats , act in zip(features, activations)
    ]

  return Sequential([x for pair in laypairs for x in pair])


emb = MLP([32]*4, [relu]*3 + [softmax])
inf = MLP([32]*4, [relu]*3 + [id])

embps = emb.init(PRNGKey(0), numpy.ones((1, 4)))
infps = inf.init(PRNGKey(1), numpy.ones((1, 32)))

params = { "embed" : embps , "infer" : infps }

ds = deepset(emb, inf)

arr = genfromtxt("../top.csv", delimiter=",", skip_header=1)

events = awkward.run_lengths(arr[:,0])

arr = awkward.unflatten(arr, events)

arr = \
  awkward.fill_none \
  ( awkward.pad_none( arr , 8 , clip=True , axis=1 )
  , [999]*6
  , axis=1
  )

arr = awkward.to_regular(arr).to_numpy().astype(numpy.float32)[:,:,1:]

mask = onp.any(arr == 999, axis=2)

print(arr.shape)
print(mask.shape)
# TODO
# batching and training
