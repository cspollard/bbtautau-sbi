from flax.linen import Module, Dense, Sequential, relu, softmax
from jax.random import PRNGKey
import jax.numpy as numpy
from jax import Array
import optax
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

print(ds(params, numpy.ones((2, 4, 4)), numpy.ones((2, 4))))

