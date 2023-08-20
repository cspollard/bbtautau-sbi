from flax.linen import Module, Dense, Sequential, relu, softmax
from jax.random import PRNGKey
import jax.numpy as numpy
from jax import Array
import optax

import awkward
from numpy import genfromtxt
from numpy.lib.recfunctions import structured_to_unstructured
import numpy as onp

from einops import repeat, rearrange, reduce


def id(xs):
  return xs


def MLP(features, activations):
  laypairs = \
    [ [ Dense(feats) , act ] \
      for feats , act in zip(features, activations)
    ]

  return Sequential([x for pair in laypairs for x in pair])


perjet = MLP([32]*4, [relu]*3 + [softmax])
perevt = MLP([32]*4, [relu]*3 + [softmax])
inference = MLP([32]*4, [relu]*3 + [id])


params = \
  { "perjet" : perjet.init(PRNGKey(0), numpy.ones((1, 5)))
  , "perevt" : perevt.init(PRNGKey(0), numpy.ones((1, 32)))
  , "inference" : inference.init(PRNGKey(1), numpy.ones((1, 32)))
  }


def forward(params, inputs, mask):
  batchsize = inputs.shape[0]
  nevt = inputs.shape[1]
  njets = inputs.shape[2]
  nfeats = inputs.shape[3]

  flattened = rearrange(inputs, "b e j f -> (b e j) f")
  unflattened = perjet.apply(params["perjet"], flattened)
  unflattened = rearrange(unflattened, "(b e j) f -> b e j f", b=batchsize, e=nevt)

  expandedmask = repeat(mask, "b e j -> b e j f", f=unflattened.shape[3])

  summed = reduce(unflattened * expandedmask, "b e j f -> b e f", "sum")

  flattened = rearrange(summed, "b e f -> (b e) f")
  unflattened = perevt.apply(params["perevt"], flattened)
  unflattened = rearrange(unflattened, "(b e) f -> b e f", b=batchsize)

  summed = reduce(unflattened, "b e f -> b f", "sum")

  return inference.apply(params["inference"], summed)


arr = genfromtxt("../top.csv", delimiter=",", skip_header=1)

events = awkward.run_lengths(arr[:,0])

arr = awkward.unflatten(arr, events)

arr = \
  awkward.fill_none \
  ( awkward.pad_none( arr , 8 , clip=True , axis=1 )
  , [999]*6
  , axis=1
  )

# need to smear b-tagging and tau id
arr = awkward.to_regular(arr).to_numpy().astype(numpy.float32)[:,:,1:]

mask = onp.any(arr == 999, axis=2)
mask = repeat(mask, "e j -> b e j", b=1)
arr = repeat(arr, "e j f -> b e j f", b=1)

print(forward(params, arr, mask))
