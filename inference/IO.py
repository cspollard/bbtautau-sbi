
import jax.numpy as numpy
import awkward
from numpy import genfromtxt

from mixjax import mixturedict

array = numpy.array

def readarr(xsectimeslumi, fname, maxjets):
  arr = genfromtxt(fname, delimiter=",", skip_header=1)
  
  events = awkward.run_lengths(arr[:,0])

  jets = awkward.unflatten(arr[:,1:7], events)

  # TODO here we read in the weights
  wgts = awkward.unflatten(arr[:,7:], events)

  jets = \
    awkward.fill_none \
    ( awkward.pad_none( jets , maxjets , clip=True , axis=1 )
    , [999]*6
    , axis=1
    )

  # TODO is this correct?
  jets = awkward.to_regular(jets).to_numpy().astype(numpy.float32)

  mask = numpy.any(jets != 999, axis=2)

  # divide momenta by 20 GeV (just a good normalization guess)
  jets[:,:,2:5] = jets[:,:,2:5] / 20

  l = len(mask)
  weights = wgts / l

  # problem seems to be here.
  return \
    mixturedict \
    ( { "events" : array(arr), "jetmasks" : array(mask) }
    , array(weights)
    )