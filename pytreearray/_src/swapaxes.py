import jax
import jax.numpy as jnp
from functools import partial, reduce

from .util import amap, _treedefs_compose, _swaptuple
from . import core


def _swapaxes(x, axes, i, j):

    i, j = min(i, j), max(i, j)

    def _prod(x):
        if len(x) == 0:
            return 0
        return reduce(lambda a, b: a * b, x)

    axes_before = _prod(axes[:i])
    start_i = axes_before
    axes_i = axes[i]
    axes_middle = _prod(axes[i + 1 : j])
    start_j = axes_before + axes_i + axes_middle
    axes_j = axes[j]
    axes_after = _prod(axes[j + 1 :])

    ax_pos_i = tuple(range(start_i, start_i + axes_i))
    ax_pos_j = tuple(range(start_j, start_j + axes_j))

    start_j_new = axes_before
    start_i_new = axes_before + axes_j + axes_middle
    ax_pos_i_new = tuple(range(start_i_new, start_i_new + axes_i))
    ax_pos_j_new = tuple(range(start_j_new, start_j_new + axes_j))

    return jnp.moveaxis(x, ax_pos_i + ax_pos_j, ax_pos_i_new + ax_pos_j_new)


def swapaxes(pt, i, j):
    tree = amap(partial(_swapaxes, i=i, j=j), pt.tree, pt.axes)

    tree_flat, _ = jax.tree_flatten(tree)
    new_treedefs = _swaptuple(pt.treedefs, i, j)
    old_axes = pt.axes

    new_axes = _swaptuple(old_axes, i, j)
    treedef = _treedefs_compose(*new_treedefs)
    tree = treedef.unflatten(tree_flat)

    return core.PyTreeArray(tree, new_treedefs, new_axes)
