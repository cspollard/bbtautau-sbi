
from matplotlib.figure import Figure
import jax.numpy as numpy


def validplot(prefix, pois, predicts, poibins):
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

  plt.hist2d(pois, mus, bins=poibins)
  fig.savefig("%s-pois.png" % prefix)

  return

