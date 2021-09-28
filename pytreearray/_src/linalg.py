import jax
import jax.numpy as jnp

from .numpy import abs as _abs, sum as _sum
from .core import PyTreeArray


def norm(a, ord=None, axis=None):
    if isinstance(a, PyTreeArray):
        if ord is not None or axis is not None:
            raise NotImplementedError
        return jax.lax.sqrt(_sum(_abs(a) ** 2))
    else:
        return jnp.norm(a, ord, axis)
