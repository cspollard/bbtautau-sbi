from flax.linen import Module, Dense, Sequential, relu, softmax
from jax.random import PRNGKey
import jax.numpy as numpy
import jax.random as random

import awkward
from numpy import genfromtxt
import numpy as onp

from einops import repeat, rearrange, reduce


def arr(xs):
  return numpy.array(xs)


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


def forward(params, inputs, evtmasks, jetmasks):
  batchsize = inputs.shape[0]
  nevt = inputs.shape[1]
  njets = inputs.shape[2]
  nfeats = inputs.shape[3]

  flattened = rearrange(inputs, "b e j f -> (b e j) f")
  unflattened = perjet.apply(params["perjet"], flattened)
  unflattened = rearrange(unflattened, "(b e j) f -> b e j f", b=batchsize, e=nevt)

  expandedjetmasks = \
    repeat(jetmasks, "b e j -> b e j f", f=unflattened.shape[3])

  summed = reduce(unflattened * expandedjetmasks, "b e j f -> b e f", "sum")

  flattened = rearrange(summed, "b e f -> (b e) f")
  unflattened = perevt.apply(params["perevt"], flattened)
  unflattened = rearrange(unflattened, "(b e) f -> b e f", b=batchsize)

  expandedevtmasks = \
    repeat(evtmasks, "b e -> b e f", f=unflattened.shape[2])

  # here we also need an event mask...
  summed = reduce(unflattened * expandedevtmasks, "b e f -> b f", "sum")

  return inference.apply(params["inference"], summed)


def readarr(fname):
  arr = genfromtxt(fname, delimiter=",", skip_header=1)

  events = awkward.run_lengths(arr[:,0])

  arr = awkward.unflatten(arr, events)

  arr = \
    awkward.fill_none \
    ( awkward.pad_none( arr , 8 , clip=True , axis=1 )
    , [999]*6
    , axis=1
    )

  # TODO
  # need to smear b-tagging and tau id
  arr = awkward.to_regular(arr).to_numpy().astype(numpy.float32)[:,:,1:]

  mask = numpy.any(arr == 999, axis=2)

  return arr, mask


datasets = { k : readarr(k + ".csv") for k in [ "top" , "HH" ] }


# lams : (b, e)
def sample(k, lams, maxn):
  b , e = lams.shape

  k1 , k2 , k3 = random.split(k, 3)

  # lamtot : (b,)
  lamtot = reduce(lams, "b w -> b", "sum")

  bernoulliprobs = repeat(lamtot / maxn, "b -> b m", m=maxn)

  # mask : (b, maxn)
  mask = random.bernoulli(k2, bernoulliprobs)

  # ps : (b, e)
  ps = lams / repeat(lamtot, "b -> b e", e=e)

  # tmpidxs : (b, e)
  tmpidxs = numpy.arange(e)

  # TODO
  # this is probably very slow.

  # idxs : (b , maxn)
  idxs = []
  nextk = k3
  for ib in range(b):
    thisk , nextk = random.split(nextk)
    idxs.append(random.choice(thisk, tmpidxs, shape=(maxn,), p=ps[ib]))

  tmp = numpy.stack(idxs, axis=0) 
  return tmp , mask


ttlams = 100 / len(datasets["top"][0]) * numpy.ones((len(datasets["top"][0]),))
hhlams = 10 / len(datasets["HH"][0]) * numpy.ones((len(datasets["HH"][0]),))
evts = numpy.concatenate([ datasets["top"][0] , datasets["HH"][0] ])
masks = numpy.concatenate([ datasets["top"][1] , datasets["HH"][1] ])


def appmu(k, mus, maxn):
  b = mus.shape[0]
  tmptt = repeat(ttlams, "e -> b e", b=b)
  tmphh = repeat(hhlams, "e -> b e", b=b)
  mus = repeat(mus, "b -> b e", e=hhlams.shape[0])

  lams = numpy.concatenate([ tmptt , mus * tmphh ], axis=1)

  return sample(k, lams, maxn)


evtidxs , evtmasks = appmu(PRNGKey(0), arr([1, 2, 3]), 256)

batch , jetmasks = evts[evtidxs], masks[evtidxs]

print(forward(params, batch, evtmasks, jetmasks))
