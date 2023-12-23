
import jax.numpy as numpy
import awkward
from numpy import genfromtxt
import einops

from mixjax import mixturedict

array = numpy.array

def readarr(fname, maxjets):
  arr = genfromtxt(fname, delimiter=",", skip_header=1)
  
  events = awkward.run_lengths(arr[:,0])

  jets = awkward.unflatten(arr[:,1:7], events)

  # take the first set of weights for each event
  wgts = awkward.unflatten(arr[:,7:], events)[:,0]

  jets = \
    awkward.fill_none \
    ( awkward.pad_none( jets , maxjets , clip=True , axis=1 )
    , [999]*6
    , axis=1
    )

  jets = awkward.to_regular(jets).to_numpy().astype(numpy.float32)

  mask = numpy.any(jets != 999, axis=2)

  # TODO
  # only _very nearly_ correct.
  weights = awkward.to_regular(wgts).to_numpy().astype(numpy.float32)

  l = weights.shape[0]
  weights = weights / l
  if weights.shape[1] != 3:
    weights = einops.repeat(weights, "e w -> e (3 w)")

  return \
    mixturedict \
    ( { "events" : array(jets), "jetmasks" : array(mask) }
    , array(weights)
    )