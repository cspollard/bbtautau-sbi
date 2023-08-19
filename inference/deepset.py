from jax import Array, jit
from flax.linen import Module
from einops import rearrange, reduce, repeat

def deepset(embed : Module, infer : Module):

  @jit
  def f(params, inputs, mask):
    batchsize = inputs.shape[0]

    flattened = rearrange(inputs, "b s f -> (b s) f")
    embedded = embed.apply(params["embed"], flattened)
    unflattened = rearrange(embedded, "(b s) f -> b s f", b=batchsize)

    expandedmask = repeat(mask, "b s -> b s f", f=unflattened.shape[2])

    summed = reduce(unflattened * expandedmask, "b s f -> b f", "sum")

    return infer.apply(params["infer"], summed)

  return f

