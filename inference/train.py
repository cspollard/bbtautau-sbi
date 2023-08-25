from jax.random import PRNGKey
import jax.numpy as numpy
import jax.random as random
import jax

from flax.linen import Module, Dense, Sequential, LayerNorm, relu, softmax

import optax

import awkward
from numpy import genfromtxt

from einops import repeat, rearrange, reduce

from tqdm import tqdm


# how many jets / evt
MAXJETS = 8
# how many evts / dataset
MAXEVTS = 700

NEPOCHS = 50
NBATCHES = 128
BATCHSIZE = 64
LR = 1e-3
NVALID = 500

# how many datasets in the test
NTEST = 1024


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


perjet = MLP([64]*6 , [relu]*5 + [softmax])
perevt = MLP([64]*6 , [relu]*5 + [softmax])
inference = MLP([64]*6 + [2] , [relu]*6 + [id])


params = \
  { "perjet" : perjet.init(PRNGKey(0), numpy.ones((1, 5)))
  , "perevt" : perevt.init(PRNGKey(0), numpy.ones((1, 64)))
  , "inference" : inference.init(PRNGKey(1), numpy.ones((1, 64)))
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

  arr = \
    awkward.fill_none \
    ( awkward.pad_none( arr , MAXJETS , clip=True , axis=1 )
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

validdata = \
  { k : (v[0][:NVALID] , v[1][:NVALID]) for k , v in datasets.items() }

traindata = \
  { k : (v[0][NVALID:] , v[1][NVALID:]) for k , v in datasets.items() }


# lams : (b, e)
def sample(k, lams, maxn):
  b , e = lams.shape

  k1 , k2 , k3 = random.split(k, 3)

  # lamtot : (b,)
  lamtot = reduce(lams, "b w -> b", "sum")
  ns = repeat(random.poisson(k2, lamtot), "b -> b e", e=maxn)
  idxs = repeat(numpy.arange(maxn), "e -> b e", b=b)

  # mask : (b, maxn)
  mask = idxs < ns

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


def appparams(knext, pois, nps, maxn, dataset):

  evts = numpy.concatenate([ dataset["top"][0] , dataset["HH"][0] ])
  masks = numpy.concatenate([ dataset["top"][1] , dataset["HH"][1] ])
  ttlams = 500 / len(dataset["top"][0]) * numpy.ones((len(dataset["top"][0]),))
  hhlams = 1 / len(dataset["HH"][0]) * numpy.ones((len(dataset["HH"][0]),))

  b = pois.shape[0]
  tmptt = repeat(ttlams, "e -> b e", b=b)

  # currently:
  # np[:,0] is the top norm uncertainty
  # no other uncerts.
  ttmus = repeat(nps, "b -> b e", e=ttlams.shape[0])

  tmphh = repeat(hhlams, "e -> b e", b=b)
  mus = repeat(pois, "b -> b e", e=hhlams.shape[0])

  lams = numpy.concatenate([ ttmus * tmptt , mus * tmphh ], axis=1)

  idxs , evtmasks = sample(knext, lams, maxn)
  batch , jetmasks = evts[idxs] , masks[idxs]
  return batch , evtmasks, jetmasks

# @jax.jit
def buildbatch(k, pois, nps, dataset):
  evtidxs , evtmasks = appparams(k, pois, nps, MAXEVTS, dataset)
  return batch , evtmasks , jetmasks


def prior(knext, b):
  k , knext = splitkey(knext)
  pois = 50 * random.uniform(k, shape=(b,))
  nps = relu(1 + 0.1 * random.normal(k, shape=(b,)))
  return pois , nps


def testprior(k, b):
  return 25 + 50 * random.uniform(k, shape=(b,))


####################

@jax.jit
def loss(outputs, pois):
  means = numpy.exp(outputs[:,0])
  logsigmas = outputs[:,1]

  chi = (means - pois) / numpy.exp(logsigmas)

  return (0.5 * chi * chi + logsigmas).mean()

@jax.jit
def runloss(params, batch, evtmasks, jetmasks, pois):
  fwd = forward(params, batch, evtmasks, jetmasks)
  return loss(fwd, pois)


@jax.jit
def step(params, opt_state, batch, evtmasks, jetmasks, pois):
  loss_value, grads = \
    jax.value_and_grad(runloss)(params, batch, evtmasks, jetmasks, pois)

  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)
opt_state = optimizer.init(params)

knext = PRNGKey(10)

k , knext = splitkey(knext)
testpois , testnps = prior(k, NTEST)

k , knext = splitkey(knext)
testbatch , testevtmasks , testjetmasks = \
  appparams(k, testpois, testnps, MAXEVTS, validdata)

for epoch in range(NEPOCHS):
  for _ in tqdm(range(NBATCHES)):
    k, knext = splitkey(knext)
    pois , nps = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    batch , evtmasks , jetmasks = appparams(k, pois, nps, MAXEVTS, traindata)

    params, opt_state, loss_value = \
      step(params, opt_state, batch, evtmasks, jetmasks, pois)


  outs = numpy.exp(forward(params, testbatch, testevtmasks, testjetmasks))

  diff = outs[:,0] - testpois
  pull = diff / outs[:,1]
  print("nevt, pois, and sample outputs")
  print(reduce(testevtmasks[:5], "b e -> b", "sum"))
  print(testnps[:5])
  print(testpois[:5])
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
  print("end epoch %02d" % epoch)
  print()

print()
print("end of training")
print()
print("nevt, pois, and sample outputs")
print(reduce(testevtmasks, "b e -> b", "sum"))
print(testnps)
print(testpois)
print(outs[:,0])
print(outs[:,1])
print()
print("sample diffs")
print(diff)
print()
print("sample pulls")
print(pull)
print()
