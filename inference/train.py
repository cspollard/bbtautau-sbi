from jax.random import PRNGKey as key
import jax.numpy as numpy
import jax.random as random
import jax

from flax.linen import Dense, Sequential, relu, softmax
from flax.training import train_state, checkpoints

import optax

from einops import repeat, rearrange, reduce

from tqdm import tqdm

from mixjax import randompartition, reweight, concat, mixturedict

from IO import readarr
from validplots import validplot

# how many jets / evt
MAXJETS = 8
# how many evts / dataset
MAXEVTS = 2048

NJETNODES = 32
NJETLAYERS = 6
NEVENTNODES = 64
NEVENTLAYERS = 6
NINFNODES = 64
NINFLAYERS = 4

NEPOCHS = 500
NBATCHES = 256
BATCHSIZE = 64
LR = 1e-3
MAXMU = 5

# how many MC events should be allocated for the validation sample
VALIDFRAC = 0.3

# how many datasets in the valid sample
NVALIDBATCHES = 1024

# checkpoints
CKPTDIR = './checkpoints'

# luminosity in 1/pb
LUMI = 10e3

bkgs = [ "top" , "ZH" , "higgs" , "DYbb" ]
sigs = [ "HH" ]


# some useful aliases
split = random.split
stack = numpy.stack


def id(xs):
  return xs


def MLP(features, activations):
  laypairs = \
    [ [ Dense(feats) , act ] \
      for feats , act in zip(features, activations)
    ]

  return Sequential([x for pair in laypairs for x in pair])


perjet = MLP([NJETNODES]*NJETLAYERS , [relu]*(NJETLAYERS-1) + [softmax])
perevt = MLP([NEVENTNODES]*NEVENTLAYERS , [relu]*(NEVENTLAYERS-1) + [softmax])
inference = MLP([NINFNODES]*NINFLAYERS + [2] , [relu]*NINFLAYERS + [id])


params = \
  { "perjet" : perjet.init(key(0), numpy.ones((1, 6)))
  , "perevt" : perevt.init(key(0), numpy.ones((1, NJETNODES)))
  , "inference" : inference.init(key(1), numpy.ones((1, NEVENTNODES)))
  }


@jax.jit
def forward(params, inputs, evtmasks, jetmasks):
  batchsize = inputs.shape[0]
  nevt = inputs.shape[1]

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


def minv(p3, mask=None):
  if mask is not None:
    mask = repeat(mask, "e p -> e p x", x=3)
    p3 = p3 * mask


  E = numpy.sqrt(reduce(p3 * p3, "e p x -> e p", "sum"))
  Etot = reduce(E, "e p -> e", "sum")
  ptot = reduce(p3, "e p x -> e x", "sum")
  m2 = Etot * Etot - reduce(ptot * ptot, "e x -> e", "sum")

  return numpy.sqrt(m2)


def ands(bools):
  ret = bools[0]
  for i in range(1, len(bools)):
    ret = numpy.logical_and(ret, bools[i])

  return ret


def select(samp):
  jets = samp.allsamples()["events"]
  mask = samp.allsamples()["jetmasks"]

  bjets = ands([mask , jets[:,:,0] == 1])
  taus = ands([mask , jets[:,:,2] == 1 , numpy.logical_not(bjets)])


  # scale = repeat(mask , "e j -> e j x", x=3) * 0.05

  # # normalize to 20 GeV
  # jets = jets.at[:,:,3:6].set(jets[:,:,3:6] * scale)

  # 2 b-jets and 2 taus
  sel = \
    numpy.logical_and \
    ( reduce( bjets , "e j -> e" , "sum" ) == 2
    , reduce( taus ,  "e j -> e" , "sum" ) == 2
    )

  # how to get just the taus and bjets now that we have their indices?
  jets = jets[sel]
  taus = taus[sel]
  bjets = bjets[sel]
  btau = numpy.logical_or(bjets, taus)
  mask = mask[sel]
  weights = samp.weights[sel]

  highmass = minv(jets[:,:,3:], ands([mask , btau])) > 200
  bmass = minv(jets[:,:,3:], ands([mask , bjets]))
  taumass = minv(jets[:,:,3:], ands([mask , taus]))

  sel = ands([highmass, bmass > 75, bmass < 150, taumass > 50, taumass < 125])

  # how to get just the taus and bjets now that we have their indices?
  jets = jets[sel]
  mask = mask[sel]
  weights = weights[sel]

  return \
    mixturedict \
    ( {"events" : jets, "jetmasks" : mask}
    , weights
    )


allsamples = \
  { k : \
      randompartition \
      ( key(0)
      , select(readarr(k + ".csv", maxjets=MAXJETS))
      , VALIDFRAC
      , compensate=True
      )
    for k in bkgs + sigs
  }


validsamps = { k : m[0] for k , m in allsamples.items() }
trainsamps = { k : m[1] for k , m in allsamples.items() }


print("done reading in samples")
print()


# pois: HH mu
# nps: mu for different bkgs
def generate(knext, pois, nps, samps):
  procs = []
  for prock in samps:
    tmp = samps[prock]

    if prock in nps:
      tmp = reweight(lambda x: LUMI*nps[prock], tmp)
    if prock == "HH":
      tmp = reweight(lambda x: LUMI*pois, tmp)

    procs.append(tmp)

  tmp = concat(procs)
  tmp.weights = tmp.weights[:,0]
  # print(numpy.sum(tmp.weights))

  return tmp.sample(knext, MAXEVTS)


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
    for p in bkgs:
      k , knext = split(knext)
      d[p] = relu(1 + 0.00005 * random.normal(k, shape=(1,)))

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
def step(opt_state, batch, evtmasks, jetmasks, pois):
  loss_value, grads = \
    jax.value_and_grad(runloss) \
      (opt_state.params, batch, evtmasks, jetmasks, pois)

  new_opt_state = opt_state.apply_gradients(grads=grads)

  return new_opt_state , loss_value

####################

sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)

print("setting up training state")
print()

opt_state = \
  train_state.TrainState.create \
  ( apply_fn=forward
  , params=params
  , tx=optimizer
  )


print("building validation sample")
print()

knext = key(10)

k , knext = split(knext)
validpois , validnps = prior(k, NVALIDBATCHES)

k , knext = split(knext)
validbatch , validevtmasks , validjetmasks = \
  buildbatch(k, validpois, validnps, validsamps)

print("building test sample")
print()

k , knext = split(knext)
testpois , testnps = prior(k, NVALIDBATCHES)

k , knext = split(knext)
testbatch , testevtmasks , testjetmasks = \
  buildbatch(k, testpois, testnps, trainsamps)



for epoch in range(NEPOCHS):
  print("start epoch %02d" % epoch)
  print()


  # TODO
  # saving checkpoints
  # if not (epoch % 10):
  #   checkpoints.save_checkpoint(ckpt_dir=CKPTDIR, target=opt_state, step=epoch//10)

  if epoch == 0:
    print("JIT may take some time during the first batch...")

  print()

  for _ in tqdm(range(NBATCHES)):
    k, knext = split(knext)
    pois , nps = prior(k, BATCHSIZE)
    k, knext = split(knext)
    batch , evtmasks , jetmasks = buildbatch_train(k, pois, nps)

    opt_state, loss_value = \
      step(opt_state, batch, evtmasks, jetmasks, pois)


  outs = numpy.exp(forward(opt_state.params, testbatch, testevtmasks, testjetmasks))

  validplot("figs/test-%02d" % epoch, testpois, outs, numpy.mgrid[0:MAXMU:25j])

  outs = numpy.exp(forward(opt_state.params, validbatch, validevtmasks, validjetmasks))

  validplot("figs/valid-%02d" % epoch, validpois, outs, numpy.mgrid[0:MAXMU:25j])

  k, knext = split(knext)
  idxs = random.choice(k, numpy.arange(NVALIDBATCHES), shape=(5,))

  diff = outs[:,0] - validpois
  pull = diff / outs[:,1]
  print("nevt, pois, posterior mu, posterior sigma")
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
print("nevt, pois, and outputs + uncertainties")
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
