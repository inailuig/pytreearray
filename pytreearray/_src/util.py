import jax
import jax.numpy as jnp
from functools import reduce


################################################################################
# tuple stuff
################################################################################


def _cumsum(x):
    # cumsum of a tuple
    return reduce(lambda c, x_: c + (c[-1] + x_,) if c else (x_,), x, ())


def _flatten(t):
    # flatten a tuple of tuples
    return reduce(lambda x, y: x + y, t)


def _swaptuple(t, i, j):
    if i == j:
        return t
    i, j = min(i, j), max(i, j)
    return t[:i] + t[j : j + 1] + t[i + 1 : j] + t[i : i + 1] + t[j + 1 :]


_arr_treedef = jax.tree_structure(jnp.zeros(0))  # TODO proper way to get * ??

################################################################################
# tree stuff
################################################################################


def _treedefs_compose(*treedefs):
    return reduce(lambda s1, s2: s1.compose(s2), treedefs)


class leaf_tuple(tuple):
    # a tuple which jax thinks its a leaf and not a subtree
    # TODO better way?
    pass


def ndim_multiply_outer(t1, t2):
    # np.multiply.outer
    return jax.tree_map(lambda l1: jax.tree_map(lambda l2: leaf_tuple(l1 + l2), t2), t1)


def build_ndim_tree(*shapedefs):
    shapedefs_ = jax.tree_map(lambda x: leaf_tuple((x,)), shapedefs)
    return reduce(ndim_multiply_outer, shapedefs_)


def amap(f, tree, axes_trees):
    # f gets array and the sizes of the axes
    axes_tree = build_ndim_tree(*axes_trees)
    return jax.tree_multimap(f, tree, axes_tree)


def size_multiply_outer(t1, t2):
    # np.multiply.outer
    return jax.tree_map(lambda l1: jax.tree_map(lambda l2: l1 * l2, t2), t1)


def build_size_tree(*shapedefs):
    return reduce(size_multiply_outer, shapedefs)


def tree_allclose(t1, t2):
    return jax.tree_structure(t1) == jax.tree_structure(t2) and jax.tree_util.tree_reduce(lambda x, y: x and y, jax.tree_multimap(jnp.allclose, t1, t2))
