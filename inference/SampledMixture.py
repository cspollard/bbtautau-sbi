from typing import Callable, Any
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
  def __init__ \
    ( self
    , samples : Callable[[Any], Any]
    , weights : jax.Array
    ) :

    self.count = weights.shape[0]
    self.idxs = numpy.arange(self.count)

    self.samples = samples
    self.weights = weights

    return


  def allsamples(self):
    return self.samples(self.idxs)


  def sample(self, knext, maxn):
    lam = self.weights.sum()

    k , knext = split(knext)
    idxs = random.choice(k, self.count, p=self.weights / lam, shape=(maxn,))

    n = random.poisson(knext, lam)
    mask = numpy.arange(maxn) < n

    return self.samples(idxs), mask


def bootstrap \
  ( k : random.KeyArray
  , mix : SampledMixture
  ) -> SampledMixture :

  l = mix.weights.shape[0]
  idxs = random.choice(k, l, shape=(n,))

  return mix[idxs]


def reweight \
  ( f : Callable[[Any], jax.Array]
  , mix : SampledMixture
  ) -> SampledMixture :

  idxs = numpy.arange(mix.count)

  return SampledMixture(mix.samples, f(mix.samples(idxs)) * mix.weights)


def alter \
  ( f : Callable[[Any], Any]
  , mix : SampledMixture
  ) -> SampledMixture :

  return SampledMixture(lambda idxs: f(mix.samples(idxs)), mix.weights)


def join \
  ( l : Callable[[jax.Array], Any]
  , r : Callable[[jax.Array], Any]
  , lenl : int
  ) -> Callable[[jax.Array], Any] :

  def f(idxs : jax.Array) -> Any:
    test = idxs < lenl
    # TODO
    # lots of unnecessary zero accesses...
    lidxs = numpy.where(test, idxs, 0)
    ridxs = numpy.where(test, 0, idxs - lenl)

    return numpy.where(test, l(lidxs), r(ridxs))

  return f


def concat2 \
  ( m1 : SampledMixture
  , m2 : SampledMixture
  ) -> SampledMixture :

  samps = join(m1.samples, m2.samples, m1.count)
  return SampledMixture(samps, numpy.concatenate([m1.weights, m2.weights]))


def concat \
  ( ms : list[SampledMixture]
  ) -> SampledMixture :

  assert len(ms) > 0

  tmp = ms[0]

  for m in ms[1:]:
    tmp = concat2(tmp, m)

  return tmp


def mix \
  ( ms : list[tuple[SampledMixture, float]]
  ) -> SampledMixture :

  ms = [ reweight(lambda x: w, m) for (m , w) in ms ]

  return concat(ms)


if __name__ == '__main__':
  test = SampledMixture(lambda idx: numpy.arange(10)[idx], numpy.ones(10))
  test = concat([test, test])
  test = mix([(test, 0.5), (test, 0.5)])
  test = alter(lambda x: x * 2, test)
  test = reweight(lambda x: 1 + x * 0.005, test)
  print(test.weights)

  out, mask = test.sample(key(0), 50)
  print(mask.sum(), out)

  out, mask = test.sample(key(1), 50)
  print(mask.sum(), out)

  out, mask = test.sample(key(2), 50)
  print(mask.sum(), out)
