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

# TODO
# reindex is really out of place... could make it more "local" in concat.

# samples : a function from indices to Any
# reindex : a way to conditionally join the outcomes of samples
# weights : a corresponding array of weights that are used for sampling
Indexed = Callable[[jax.Array], Any]
class SampledMixture:
  def __init__ \
    ( self
    , samples : Indexed
    , reindex : Callable[[Indexed, Indexed, int], Indexed]
    , weights : jax.Array
    ) :

    self.count = weights.shape[0]
    self.idxs = numpy.arange(self.count)

    self.samples = samples
    self.weights = weights
    self.reindex = reindex

    return


  def allsamples(self):
    return self.samples(self.idxs)


  def __getitem__(self, idxs):
    return \
      SampledMixture \
      ( lambda jdxs: self.samples(idxs[jdxs])
      , self.reindex
      , self.weights[idxs]
      )


  def sample(self, knext, maxn):
    lam = self.weights.sum()

    k , knext = split(knext)
    idxs = random.choice(k, self.count, p=self.weights / lam, shape=(maxn,))

    n = random.poisson(knext, lam)
    # probably too slow to assert?
    # assert n <= maxn
    mask = numpy.arange(maxn) < n

    return self.samples(idxs), mask


def randompartition \
  ( k : random.KeyArray
  , mix : SampledMixture
  , frac : float
  ) -> SampledMixture :

  cutoff = int(frac * mix.count)
  perm = random.permutation(k, mix.count)

  return \
    ( mix[perm[:cutoff]]
    , mix[perm[cutoff:]]
    )


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
    , mix.reindex
    , f(mix.samples(idxs)) * mix.weights
    )


def alter \
  ( f : Callable[[Any], Any]
  , mix : SampledMixture
  ) -> SampledMixture :

  return \
    SampledMixture \
    ( lambda idxs: f(mix.samples(idxs))
    , mix.reindex
    , mix.weights
    )


def concat2 \
  ( m1 : SampledMixture
  , m2 : SampledMixture
  ) -> SampledMixture :

  samps = m1.reindex(m1.samples, m2.samples, m1.count)

  return \
    SampledMixture \
    ( samps
    , m1.reindex
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


def reindexlist \
  ( l : indexedlist
  , r : indexedlist
  , lenl : int
  ) -> indexedlist :

  def f(idxs):
    cond = idxs < lenl
    idxsl = numpy.where(cond, idxs, 0)
    idxsr = numpy.where(cond, 0, idxs - lenl)
    return [ numpy.where(cond, x, y) for (x, y) in zip(l(idxsl), r(idxsr)) ]

  return f


def indexdict(xs) -> indexedlist :

  def f(idxs : jax.Array) :
    return { k : x[idxs] for k , x in xs.items() }

  return f


def reindexdict \
  ( l
  , r
  , lenl
  ) :

  def f(idxs):
    cond = idxs < lenl
    idxsl = numpy.where(cond, idxs, 0)
    idxsr = numpy.where(cond, 0, idxs - lenl)

    ld = l(idxsl)
    rd = r(idxsr)

    # would prefer this, but it's probably slow.
    # assert ld.keys() == rd.keys()

    # really annoying I need to broadcast here via transposition?
    return \
      { k : numpy.where(cond, ld[k].T, rd[k].T).T
        for k in ld
      }

  return f


def mixturedict \
  ( samplesdict
  , weights
  ) :
  return SampledMixture(indexdict(samplesdict), reindexdict, weights)


if __name__ == '__main__':
  test = \
    mixturedict \
    ( { "hi" : numpy.arange(10) , "bye" : numpy.arange(10) }
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

  print(len(out["hi"]))
  print(len(out["bye"]))
