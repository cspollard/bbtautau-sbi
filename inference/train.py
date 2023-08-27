from jax.random import PRNGKey as key
import jax.numpy as numpy
import jax.random as random
import jax

from flax.linen import Module, Dense, Sequential, LayerNorm, relu, softmax

import optax

import awkward
from numpy import genfromtxt

from einops import repeat, rearrange, reduce

from tqdm import tqdm

from SampledMixture \
  import mixturedict, randompartition, reweight, concat2

# how many jets / evt
MAXJETS = 8
# how many evts / dataset
MAXEVTS = 256

NEPOCHS = 50
NBATCHES = 128
BATCHSIZE = 64
LR = 1e-3

# how many MC events should be allocated for the validation sample
VALIDFRAC = 0.3

# how many datasets in the valid sample
NVALIDBATCHES = 1024


xsecs = { "top" : 100 , "HH" : 10 }


split = random.split
arr = numpy.array
stack = numpy.stack

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
  { "perjet" : perjet.init(key(0), numpy.ones((1, 5)))
  , "perevt" : perevt.init(key(0), numpy.ones((1, 64)))
  , "inference" : inference.init(key(1), numpy.ones((1, 64)))
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


def readarr(xsectimeslumi, fname):
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

  mask = numpy.any(arr != 999, axis=2)

  # divide momenta by 20 GeV
  arr[:,:,2:5] = arr[:,:,2:5] / 20

  l = len(mask)
  weights = xsectimeslumi / l * numpy.ones((l,))

  return mixturedict({ "events" : arr, "jetmasks" : mask }, weights)


allsamples = \
  { k : readarr(xsecs[k] , k + ".csv")
    for k in [ "top" , "HH" ]
  }

validsamps = allsamples
trainsamps = allsamples

# allsamples = \
#   { k : randompartition(key(0), readarr(xsecs[k] , k + ".csv"), VALIDFRAC)
#     for k in [ "top" , "HH" ]
#   }

# validsamps = { k : m[0] for k , m in allsamples.items() }
# trainsamps = { k : m[1] for k , m in allsamples.items() }

print("done reading in samples")


# pois: HH mu
# nps: tt mu
def generate(knext, pois, nps, samps):
  HH = reweight(lambda x: pois, samps["HH"])
  top = reweight(lambda x: nps, samps["top"])

  return concat2(HH, top).sample(knext, MAXEVTS)


# TODO
# I bet this is the slow bit...
# can't jit this at the moment...
# @jax.jit
def buildbatch(knext, pois, nps, samps):
  nbatch = pois.shape[0]

  batches = []
  jetmasks = []
  evtmasks = []
  for i in range(nbatch):
    k , knext = split(knext)
    batch , masks = generate(k, pois[i], nps[i], samps)
    batches.append(batch["events"])
    jetmasks.append(batch["jetmasks"])
    evtmasks.append(masks)

  return \
    stack(batches) , stack(evtmasks) , stack(jetmasks)


def prior(knext, b):
  k , knext = split(knext)
  pois = random.uniform(k, shape=(b,))
  nps = relu(1 + 0.1 * random.normal(k, shape=(b,)))
  return pois , nps


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

knext = key(10)

k , knext = split(knext)
validpois , validnps = prior(k, NVALIDBATCHES)

k , knext = split(knext)
validbatch , validevtmasks , validjetmasks = \
  buildbatch(k, validpois, validnps, validsamps)


for epoch in range(NEPOCHS):
  for _ in tqdm(range(NBATCHES)):
    k, knext = split(knext)
    pois , nps = prior(k, BATCHSIZE)
    k, knext = split(knext)
    batch , evtmasks , jetmasks = buildbatch(k, pois, nps, trainsamps)

    params, opt_state, loss_value = \
      step(params, opt_state, batch, evtmasks, jetmasks, pois)


  outs = numpy.exp(forward(params, validbatch, validevtmasks, validjetmasks))

  k, knext = split(knext)
  idxs = random.choice(k, numpy.arange(NVALIDBATCHES), shape=(5,))

  diff = outs[:,0] - validpois
  pull = diff / outs[:,1]
  print("nevt, nps, pois, posterior mu, posterior sigma")
  print(reduce(validevtmasks[idxs], "b e -> b", "sum"))
  print(validnps[idxs])
  print(validpois[idxs])
  print(outs[idxs,0])
  print(outs[idxs,1])
  print()
  print("sample diffs")
  print(diff[idxs])
  print()
  print("sample pulls")
  print(pull[idxs])
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
print("nevt, nps, pois, and outputs + uncertainties")
print(reduce(validevtmasks, "b e -> b", "sum"))
print(validnps)
print(validpois)
print(outs[:,0])
print(outs[:,1])
print()
print("sample diffs")
print(diff)
print()
print("sample pulls")
print(pull)
print()
