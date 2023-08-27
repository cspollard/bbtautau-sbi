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
  import mixturedict, randompartition, reweight, concat

from matplotlib.figure import Figure

# how many jets / evt
MAXJETS = 8
# how many evts / dataset
MAXEVTS = 200

NEPOCHS = 50
NBATCHES = 512
BATCHSIZE = 32
LR = 1e-3
MAXMU = 5

# how many MC events should be allocated for the validation sample
VALIDFRAC = 0.3

# how many datasets in the valid sample
NVALIDBATCHES = 2048


xsectimeslumis = { "top" : 100 , "ZH" : 10 , "higgs" : 10 , "HH" : 10 }


split = random.split
array = numpy.array
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
  # JES, JER, etc
  arr = awkward.to_regular(arr).to_numpy().astype(numpy.float32)[:,:,1:]

  mask = numpy.any(arr != 999, axis=2)

  # divide momenta by 20 GeV
  arr[:,:,2:5] = arr[:,:,2:5] / 20

  l = len(mask)
  weights = xsectimeslumi / l * numpy.ones((l,))

  return \
    mixturedict \
    ( { "events" : array(arr), "jetmasks" : array(mask) }
    , array(weights)
    )


allsamples = \
  { k : \
      randompartition \
      ( key(0)
      , readarr(xsectimeslumis[k], k + ".csv")
      , VALIDFRAC
      , compensate=True
      )

    for k in [ "top" , "ZH" , "higgs" , "HH" ]
  }

validsamps = { k : m[0] for k , m in allsamples.items() }
trainsamps = { k : m[1] for k , m in allsamples.items() }


print("done reading in samples")


# pois: HH mu
# nps: tt mu
def generate(knext, pois, nps, samps):
  procs = []
  for prock in samps:
    tmp = samps[prock]
    if prock in nps:
      tmp = reweight(lambda x: nps[prock], tmp)
    if prock == "HH":
      tmp = reweight(lambda x: pois, tmp)

    procs.append(tmp)

  return concat(procs).sample(knext, MAXEVTS)


# TODO
# this strategy results in an extremely slow first batch because of the for-loop
# in buildbatch. 
# could we do one pass with b = 1, then ramp up to b = BATCHSIZE?
@jax.jit
def buildbatch_train(knext, pois, nps):
  return buildbatch(knext, pois, nps, trainsamps)


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

  return stack(batches) , stack(evtmasks) , stack(jetmasks)


def prior(knext, b):
  k , knext = split(knext)
  pois = MAXMU * random.uniform(k, shape=(b,))

  nps = []
  for i in range(b):
    d = {}
    for p in [ "top" , "ZH" , "higgs" ]:
      k , knext = split(knext)
      d[p] = relu(1 + 0.3 * random.normal(k, shape=(1,)))

    nps.append(d)

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


def plot(prefix, pois, predicts):
  mus = predicts[:,0]
  sigmas = predicts[:,1]

  diffs = mus - pois
  pulls = diffs / sigmas

  fig = Figure((6, 6))
  plt = fig.add_subplot()

  bins = numpy.mgrid[-3:3:30j]

  plt.hist(pulls, bins=bins)
  fig.savefig("%s-pulls.png" % prefix)


  fig = Figure((6, 6))
  plt = fig.add_subplot()

  bins = numpy.mgrid[-3:3:30j]

  plt.hist(diffs, bins=bins)
  fig.savefig("%s-diffs.png" % prefix)


  fig = Figure((6, 6))
  plt = fig.add_subplot()

  bins = numpy.mgrid[0:MAXMU:25j]

  plt.hist2d(pois, mus, bins=bins)
  fig.savefig("%s-pois.png" % prefix)

  return


  print("nevt, nps, pois, posterior mu, posterior sigma")
  print(reduce(evtmasks[idxs], "b e -> b", "sum"))
  print(nps[idxs])
  print(pois[idxs])
  print(predicts[idxs,0])
  print(predicts[idxs,1])
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



sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)
opt_state = optimizer.init(params)

knext = key(10)

k , knext = split(knext)
validpois , validnps = prior(k, NVALIDBATCHES)

k , knext = split(knext)
validbatch , validevtmasks , validjetmasks = \
  buildbatch(k, validpois, validnps, validsamps)

k , knext = split(knext)
testpois , testnps = prior(k, NVALIDBATCHES)

k , knext = split(knext)
testbatch , testevtmasks , testjetmasks = \
  buildbatch(k, testpois, testnps, trainsamps)


for epoch in range(NEPOCHS):
  print("start epoch %02d" % epoch)

  if epoch == 0:
    print("JIT may take some time during the first batch...")

  print()

  for _ in tqdm(range(NBATCHES)):
    k, knext = split(knext)
    pois , nps = prior(k, BATCHSIZE)
    k, knext = split(knext)
    batch , evtmasks , jetmasks = buildbatch_train(k, pois, nps)

    params, opt_state, loss_value = \
      step(params, opt_state, batch, evtmasks, jetmasks, pois)


  outs = numpy.exp(forward(params, testbatch, testevtmasks, testjetmasks))

  plot("figs/test-%02d" % epoch, testpois, outs)

  outs = numpy.exp(forward(params, validbatch, validevtmasks, validjetmasks))

  plot("figs/valid-%02d" % epoch, validpois, outs)

  k, knext = split(knext)
  idxs = random.choice(k, numpy.arange(NVALIDBATCHES), shape=(5,))

  diff = outs[:,0] - validpois
  pull = diff / outs[:,1]
  print("nevt, nps, pois, posterior mu, posterior sigma")
  print(reduce(validevtmasks[idxs], "b e -> b", "sum"))
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
