from collections.abc import Callable
from einops import reduce
import jax.numpy as numpy
import jax.random as random
import jax

key = random.PRNGKey
split = random.split


# TODO
# consider just keeping track of alterations rather than in-place changes?
# e.g. provide a member alterations that is applied after sampling
# see https://hackage.haskell.org/package/mwc-probability-2.3.1/docs/System-Random-MWC-Probability.html#t:Prob

# samples : a list of arrays
# this is a list so that we can e.g. carry masks through
# weights : a corresponding array of weights that are used for sampling
class SampledMixture:
  def __init__(self, samples : list[jax.Array], weights : jax.Array):
    self.count = weights.shape[0]
    assert samples.shape[0] == self.count

    self.samples = samples
    self.weights = weights

    return


  def sample(self, knext, maxn):
    lam = self.weights.sum()

    k , knext = split(knext)
    idxs = random.choice(k, self.count, p=self.weights / lam, shape=(maxn,))
    samps = [ s[idxs] for s in self.samples ]

    n = random.poisson(knext, lam)
    mask = numpy.arange(maxn) < n

    return samps, mask



def resize \
  ( k : random.KeyArray
  , mix : SampledMixture
  , n : int
  ) -> SampledMixture :

  l = mix.weights.shape[0]
  idxs = random.choice(k, l, shape=(n,))

  return SampledMixture(mix.samples[idxs], mix.weights[idxs])


def bootstrap \
  ( k : random.KeyArray
  , mix : SampledMixture
  ) -> SampledMixture :

  l = mix.weights.shape[0]

  return resize(k, mix, l)


def reweight \
  ( f : Callable[[list[jax.Array]], jax.Array]
  , mix : SampledMixture
  ) -> SampledMixture :

  return SampledMixture(mix.samples, f(mix.samples) * mix.weights)


def alter \
  ( f : Callable[[jax.Array], jax.Array]
  , mix : SampledMixture
  ) -> SampledMixture :

  return SampledMixture(f(mix.samples), mix.weights)


def concat \
  ( ms
  ) -> SampledMixture :

  return \
    SampledMixture \
    ( numpy.concatenate([ m.samples for m in ms ])
    , numpy.concatenate([ m.weights for m in ms ])
    )


def mix \
  ( ms : list[tuple[SampledMixture, float]]
  ) -> SampledMixture :

  ms = [ reweight(lambda x: w, m) for (m , w) in ms ]

  return concatenate(ms)


