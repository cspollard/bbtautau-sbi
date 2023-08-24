from jax.random import PRNGKey
import jax.numpy as numpy
import jax.random as random
import jax

from flax.linen import Module, Dense, Sequential, LayerNorm, relu, softmax

import optax

import awkward
from numpy import genfromtxt

from einops import repeat, rearrange, reduce


NEPOCHS = 100
NBATCHES = 128
BATCHSIZE = 64
NMAX = 512
LR = 1e-3
NTEST = 1024


def splitkey(k):
  return random.split(k)

def arr(xs):
  return numpy.array(xs)

def id(xs):
  return xs


def MLP(features, norms, activations):
  laypairs = \
    [ [ Dense(feats) , norm , act ] \
      for feats , norm , act in zip(features, norms, activations)
    ]

  return Sequential([x for pair in laypairs for x in pair])


perjet = MLP([32]*4 , [id]*4 , [relu]*3 + [softmax])
perevt = MLP([32]*4 , [id]*4 , [relu]*3 + [softmax])
inference = MLP([32]*4 + [2] , [id]*5 , [relu]*4 + [id])


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

  summed = reduce(unflattened * expandedevtmasks, "b e f -> b f", "sum")

  return inference.apply(params["inference"], summed)


def readarr(fname):
  arr = genfromtxt(fname, delimiter=",", skip_header=1)
  
  events = awkward.run_lengths(arr[:,0])

  arr = awkward.unflatten(arr, events)

  # divide px, py, pz by 20 GeV
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

  # divide momenta by 20 GeV
  arr[:,:,2:5] = arr[:,:,2:5] / 20


  return arr , mask


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


ttlams = 100 / len(datasets["top"][0]) * numpy.ones((len(datasets["top"][0]),))
hhlams = 1 / len(datasets["HH"][0]) * numpy.ones((len(datasets["HH"][0]),))
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
  return 100 * random.uniform(k, shape=(b,))

def testprior(k, b):
  return 25 + 50 * random.uniform(k, shape=(b,))



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


sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)

optimizer = optax.adam(learning_rate=sched)

opt_state = optimizer.init(params)

knext = PRNGKey(10)

k , knext = splitkey(knext)
testlabels = prior(k, NTEST)
k , knext = splitkey(knext)
testidxs , testevtmasks = appmu(k, testlabels, NMAX)
testbatch , testjetmasks = evts[testidxs] , masks[testidxs]

for _ in range(NEPOCHS):
  for _ in range(NBATCHES):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    evtidxs , evtmasks = appmu(k, labels, NMAX)
    batch , jetmasks = evts[evtidxs] , masks[evtidxs]

    params, opt_state, loss_value = \
      step(params, opt_state, batch, evtmasks, jetmasks, labels)


  outs = numpy.exp(forward(params, testbatch, testevtmasks, testjetmasks))

  diff = outs[:,0] - testlabels
  pull = diff / outs[:,1]
  print("labels and sample outputs")
  print(testlabels[:5])
  print(outs[:5,0])
  print(outs[:5,1])
  print()
  print("sample diffs")
  print(diff[:5])
  print()
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
