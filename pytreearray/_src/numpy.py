import jax
import jax.numpy as jnp

from .core import PyTreeArray


# TODO generate all _elementwise functions automatically
# TODO use some form of dipatch?


def sqrt(t):
    if isinstance(t, PyTreeArray):
        return t._elementwise(jax.lax.sqrt)
    else:
        return jnp.sqrt(t)


def abs(t):
    if isinstance(t, PyTreeArray):
        return t._elementwise(jax.lax.abs)
    else:
        return jnp.abs(t)


def sum(a, axis=None):
    if isinstance(a, PyTreeArray):
        # TODO implement axis
        if axis is not None:
            raise NotImplementedError
        return jax.tree_util.tree_reduce(jnp.add, jax.tree_map(jnp.sum, a.tree))
    else:
        return jnp.sum(a, axis)
