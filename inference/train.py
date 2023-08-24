from jax.random import PRNGKey
import jax.numpy as numpy
import jax.random as random
import jax

from flax.linen import Module, Dense, Sequential, relu, softmax

import optax

import awkward
from numpy import genfromtxt
import numpy as onp

from einops import repeat, rearrange, reduce

NEPOCHS = 10
EPOCHSIZE = 128
BATCHSIZE = 64
NMAX = 512


def splitkey(k):
  return random.split(k)

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
inference = MLP([32]*4 + [2], [relu]*4 + [id])


params = \
  { "perjet" : perjet.init(PRNGKey(0), numpy.ones((1, 5)))
  , "perevt" : perevt.init(PRNGKey(0), numpy.ones((1, 32)))
  , "inference" : inference.init(PRNGKey(1), numpy.ones((1, 32)))
  }


@jax.jit
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
    thisk , nextk = splitkey(nextk)
    idxs.append(random.choice(thisk, tmpidxs, shape=(maxn,), p=ps[ib]))

  tmp = numpy.stack(idxs, axis=0) 
  return tmp , mask


ttlams = 200 / len(datasets["top"][0]) * numpy.ones((len(datasets["top"][0]),))
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


def prior(k, b):
  return 10 * random.uniform(k, shape=(b,))



####################

@jax.jit
def loss(outputs, labels):
  means = numpy.exp(outputs[:,0])
  logsigmas = outputs[:,1]

  chi = (means - labels) / numpy.exp(logsigmas)

  return (0.5 * chi * chi + logsigmas).mean()

@jax.jit
def runloss(params, batch, evtmasks, jetmasks, labels):
  fwd = forward(params, batch, evtmasks, jetmasks)
  return loss(fwd, labels)


@jax.jit
def step(params, opt_state, batch, evtmasks, jetmasks, labels):
  loss_value, grads = \
    jax.value_and_grad(runloss)(params, batch, evtmasks, jetmasks, labels)

  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(params)

knext = PRNGKey(10)

for _ in range(NEPOCHS):
  for _ in range(EPOCHSIZE):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    evtidxs , evtmasks = appmu(k, labels, NMAX)
    batch , jetmasks = evts[evtidxs] , masks[evtidxs]

    params, opt_state, loss_value = \
      step(params, opt_state, batch, evtmasks, jetmasks, labels)


  outs = numpy.exp(forward(params, batch, evtmasks, jetmasks))

  pull = (outs[:,0] - labels) / outs[:,1]
  print("sample pulls")
  print(pull[:5])
  print()
  print("mean pull")
  print(pull.mean())
  print()
  print("std pull")
  print(numpy.std(pull))
  print()
  print("end epoch")
  print()
