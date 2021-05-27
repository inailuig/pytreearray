import jax
import jax.numpy as jnp
from functools import partial, reduce


def tree_multiply_outer(t1, t2):
    # np.multiply.outer
    return jax.tree_map(lambda l1: jax.tree_map(lambda l2: jnp.outer(l1, l2), t2), t1)


def multiply_outer(pt1, pt2):
    # np.multiply.outer
    tree = tree_multiply_outer(pt1.tree, pt2.tree)
    treedefs = pt1.treedefs + pt2.treedefs
    axes = pt1.axes + pt2.axes
    return PyTreeArray(tree, treedefs, axes)
