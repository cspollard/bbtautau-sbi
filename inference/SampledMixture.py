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
Indexed = Callable[[jax.Array], Any]
class SampledMixture:
  def __init__ \
    ( self
    , samples : Indexed
    , where : Callable[[jax.Array, Indexed, Indexed], Indexed]
    , weights : jax.Array
    ) :

    self.count = weights.shape[0]
    self.idxs = numpy.arange(self.count)

    self.samples = samples
    self.weights = weights
    self.where = where

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

  return \
    SampledMixture \
    ( mix.samples
    , mix.where
    , f(mix.samples(idxs)) * mix.weights
    )


def alter \
  ( f : Callable[[Any], Any]
  , mix : SampledMixture
  ) -> SampledMixture :

  return \
    SampledMixture \
    ( lambda idxs: f(mix.samples(idxs))
    , mix.where
    , mix.weights
    )


def join \
  ( l : Indexed
  , r : Indexed
  , where : Callable[[jax.Array, Indexed, Indexed], Indexed]
  , lenl : int
  ) -> Callable[[jax.Array], Any] :

  def f(idxs : jax.Array) -> Any:
    test = idxs < lenl
    # TODO
    # lots of unnecessary zero accesses...
    # won't work with zero-len arrays.
    lidxs = numpy.where(test, idxs, 0)
    ridxs = numpy.where(test, 0, idxs - lenl)

    return where(test, l, r)(idxs)

  return f


def concat2 \
  ( m1 : SampledMixture
  , m2 : SampledMixture
  ) -> SampledMixture :

  samps = join(m1.samples, m2.samples, m1.where, m1.count)
  return \
    SampledMixture \
    ( samps
    , m1.where
    , numpy.concatenate([m1.weights, m2.weights])
    )


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


arrlist = list[jax.Array]
indexedlist = Callable[[jax.Array], arrlist]

def indexlist \
  ( xs : arrlist
  ) -> indexedlist :

  def f(idxs : jax.Array) -> arrlist :
    return [ x[idxs] for x in xs ]

  return f


def wherelist \
  ( cond : jax.Array
  , l : indexedlist
  , r : indexedlist
  ) -> indexedlist :

  def f(idxs):
    return [ numpy.where(cond, x, y) for (x, y) in zip(l(idxs), r(idxs)) ]

  return f


def indexdict(xs) -> indexedlist :

  def f(idxs : jax.Array) :
    return { k : x[idxs] for k , x in xs.items() }

  return f


def wheredict \
  ( cond : jax.Array
  , l
  , r
  ) :

  def f(idxs):
    ld = l(idxs)
    rd = r(idxs)

    # would prefer this, but it's probably slow.
    # assert ld.keys() == rd.keys()

    return \
      { k : numpy.where(cond, ld[k], rd[k]) for k in ld }

  return f


if __name__ == '__main__':
  test = \
    SampledMixture \
    ( indexdict({ "hi" : numpy.arange(10) , "bye" : numpy.arange(10) })
    , wheredict
    , numpy.ones(10)
    )

  def update(d, k, f):
    return 

  test = concat([test, test])
  test = mix([(test, 0.5), (test, 0.5)])
  test = alter(lambda d: d | { "hi" : d["hi"] * 2 }, test)
  test = reweight(lambda d: 1 + d["hi"] * 0.005, test)
  print(test.weights)

  out, mask = test.sample(key(0), 50)
  print(mask.sum(), out)

  out, mask = test.sample(key(1), 50)
  print(mask.sum(), out)

  out, mask = test.sample(key(2), 50)
  print(mask.sum(), out)
