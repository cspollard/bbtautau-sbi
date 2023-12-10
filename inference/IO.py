
import jax.numpy as numpy
import awkward
from numpy import genfromtxt

from mixjax import mixturedict

array = numpy.array

def readarr(xsectimeslumi, fname, maxjets):
  arr = genfromtxt(fname, delimiter=",", skip_header=1)
  
  events = awkward.run_lengths(arr[:,0])

  # TODO I'm not sure about this, but I think it should get the jet information.
  arr = awkward.unflatten(arr[1:7], events)

  # TODO here we read in the weights
  wgts = awkward.unflatten(arr[7:], events)

  arr = \
    awkward.fill_none \
    ( awkward.pad_none( arr , maxjets , clip=True , axis=1 )
    , [999]*6
    , axis=1
    )

  arr = awkward.to_regular(arr).to_numpy().astype(numpy.float32)[:,:,1:]

  mask = numpy.any(arr != 999, axis=2)

  # divide momenta by 20 GeV (just a good normalization guess)
  arr[:,:,2:5] = arr[:,:,2:5] / 20

  l = len(mask)
  weights = xsectimeslumi / l * numpy.ones((l,))

  return \
    mixturedict \
    ( { "events" : array(arr), "jetmasks" : array(mask) }
    , array(weights)
    )