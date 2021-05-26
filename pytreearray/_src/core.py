import jax
import jax.numpy as jnp
import jax.flatten_util
from flax import struct
from functools import partial, reduce, singledispatchmethod
from typing import Any, Sequence, Callable, Collection, Union

from functools import reduce
from operator import mul

import math

PyTree = Any
# Scalar = Union[float, int, complex]


@struct.dataclass
class PyTreeArrayT:
    tree: PyTree
    treedefs: Any = struct.field(pytree_node=False)
    axes: Any = struct.field(pytree_node=False)  # Trees with the number of axes

    @property
    def T(self):
        return self.transpose()

    def _isnd(self, n):
        assert len(self.treedefs) == len(self.axes)
        return len(self.treedefs) == n

    def _is1d(self):
        return self._isnd(1)

    def _is2d(self):
        return self._isnd(2)

    @property
    def _treedef(self):
        td = reduce(lambda s1, s2: s1.compose(s2), self.treedefs)
        assert td == jax.tree_structure(self.tree)
        return td

    def _leafdef(self, start, end=-1):
        td = reduce(lambda s1, s2: s1.compose(s2), self.treedefs[i:end])
        assert td == jax.tree_structure(self.tree)
        return td

    @property
    def treedef_l(self):
        # TODO deprecate?
        return self.treedefs[0]

    @property
    def treedef_r(self):
        # TODO deprecate?
        return self.treedefs[-1]

    @property
    def axes_l(self):
        # TODO deprecate?
        return self.axes[0]

    @property
    def axes_r(self):
        # TODO deprecate?
        return self.axes[-1]

    @property
    def ndim(self):
        n = len(self.treedefs)
        assert n == len(self.axes)
        return n

    def transpose(self):
        tree = amap(_transpose, self.tree, self.axes)
        treedefs = self.treedefs[::-1]

        tree_flat, _ = jax.tree_flatten(tree)
        treedef = _treedefs_compose(*treedefs)
        tree = treedef.unflatten(tree_flat)

        axes = self.axes[::-1]
        return PyTreeArrayT(tree, treedefs, axes)

    def tree_trans(self):
        # TODO arbitrary axes

        if self._is1d():
            return self

        assert self._is2d()
        # just transpose the treedef
        return jax.tree_transpose(self.treedef_l, self.treedef_r, self.tree)

    # TODO @singledispatchmethod
    def __add__(self, t: PyTree):
        if jnp.isscalar(t):
            res = jax.tree_map(lambda x: x + t, self.tree)
            return self.replace(tree=res)
        elif isinstance(t, PyTreeArrayT):
            return self + t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.add, self.tree, t)
            return self.replace(tree=res)

    def __rmul__(self, t):
        return self * t

    def __radd__(self, t):
        return self + t

    def __rsub__(self, t):
        return (-self) + t

    def __neg__(self):
        return (-1) * self

    def _elementwise(self, f):
        return self.replace(tree=jax.tree_map(f, self.tree))

    def __mul__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x * t)
        elif isinstance(t, PyTreeArrayT):
            # TODO check equal treedef_l and treedef_r, axes
            return self * t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.multiply, self.tree, t)
            return self.replace(tree=res)

    def __truediv__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x / t)
        elif isinstance(t, PyTreeArrayT):
            # TODO check equal treedef_l and treedef_r, axes
            return self / t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.divide, self.tree, t)
            return self.replace(tree=res)

    def __sub__(self, t: PyTree):
        if jnp.isscalar(t):
            return self._elementwise(lambda x: x - t)
        elif isinstance(t, PyTreeArrayT):
            return self - t.tree
        else:  # PyTree
            assert self._treedef == jax.tree_structure(t)
            res = jax.tree_multimap(jnp.subtract, self.tree, t)
            return self.replace(tree=res)

    def __pow__(self, t):
        assert jnp.isscalar(t)
        return self._elementwise(lambda x: x ** t)

    def __getitem__(self, *args, **kwargs):
        return self.tree.__getitem__(*args, **kwargs)

    def __matmul__(pt1, pt2):

        # TODO!!
        if not isinstance(pt2, PyTreeArrayT):
            # assume its a pytree vector
            pt2 = PyTreeArray(pt2)
        else:
            assert not pt2._isnd(0)

        assert pt1.treedef_r == pt2.treedef_l
        assert pt1.axes_r == pt2.axes_l  # TODO

        def tree_dot(t1, t2, axes_tree):
            res = jax.tree_util.tree_reduce(jax.lax.add, jax.tree_multimap(jnp.tensordot, t1, t2, axes_tree))
            return res

        pt1_1d = False
        if pt1._is1d():
            pt1 = pt2.replace(axes=(0,) + pt1.axes, treedefs=(_arr_treedef,) + pt2.treedefs)
            pt1_1d = True

        pt2_1d = False
        if pt2._is1d():
            pt2 = pt2.replace(axes=pt2.axes + (0,), treedefs=pt2.treedefs + (_arr_treedef,))
            pt2_1d = True

        is_leaf = lambda l: jax.tree_structure(l) == pt1.treedef_r
        tree = jax.tree_map(
            lambda t1: jax.tree_map(
                lambda t2: tree_dot(t1, t2, pt1.axes_r),
                pt2.tree_trans(),
                is_leaf=is_leaf,
            ),
            pt1.tree,
            is_leaf=is_leaf,
        )
        if pt1_1d and pt2_1d:
            return PyTreeArrayT(tree, (), ())
        elif pt1_1d:
            return PyTreeArrayT(tree, (pt2.treedef_r,), (pt2.axes_r,))
        elif pt2_1d:
            return PyTreeArrayT(tree, (pt1.treedef_l,), (pt1.axes_l,))
        else:
            return PyTreeArrayT(tree, (pt1.treedef_l, pt2.treedef_r), (pt1.axes_l, pt2.axes_r))

    def conjugate(self):
        return self._elementwise(jnp.conj)

    def conj(self):
        return self.conjugate()

    @property
    def imag(self):
        return self._elementwise(jnp.imag)

    @property
    def real(self):
        return self._elementwise(jnp.real)

    @property
    def H(self):
        return self.T.conj()

    def _l_map(self, f):
        return jax.tree_map(f, self.tree, is_leaf=lambda x: jax.tree_structure(x) == self.treedef_r)

    def _lr_map(self, f):
        return self._l_map(lambda r: jax.tree_map(f, r))

    def _lr_amap(self, f):
        return self._l_map(lambda r: jax.tree_multimap(f, r, self.axes_r))

    def _flatten_tensors(self):
        def _fl(x, axes):
            shape = x.shape
            start = _cumsum((0,) + axes[:-1])

            def _f(il):
                i, l = il
                return _prod(shape[i : i + l])

            new_shape = tuple(map(_f, zip(start, axes)))
            return x.reshape(new_shape)

        tree = amap(_fl, self.tree, self.axes)

        _set1 = lambda x: jax.tree_map(lambda _: 1, x)
        axes = _set1(self.axes)
        return self.replace(tree=tree, axes=axes)

    def to_dense(self):
        tree_flat = self._flatten_tensors().tree
        for i, td in enumerate(self.treedefs[::-1]):
            is_leaf = lambda l: jax.tree_structure(l) == td

            @partial(jax.vmap, in_axes=i, out_axes=i)
            def _f(x):
                return jax.flatten_util.ravel_pytree(x)[0]

            tree_flat = jax.tree_map(_f, tree_flat, is_leaf=is_leaf)
        return tree_flat

    def add_diag_scalar(self, a):
        assert self.treedef_l == self.treedef_r
        nl = self.treedef_l.num_leaves

        def _is_diag(i):
            return i % (nl + 1) == 0

        def _tree_map_diag(f, tree, is_diag):
            leaves, treedef = jax.tree_flatten(tree)
            return treedef.unflatten(f(l) if is_diag(i) else l for i, l in enumerate(leaves))

        def _add_diag_tensor(x, a):
            # TODO simpler ?
            s = x.shape
            n = x.ndim
            assert n % 2 == 0
            assert s[: n // 2] == s[n // 2 :]
            sl = s[: n // 2]
            _prod = lambda x: reduce(mul, x, 1)
            il = jnp.unravel_index(jnp.arange(_prod(sl)), sl)
            i = il + il
            return jax.ops.index_add(x, i, a)

        tree = _tree_map_diag(partial(_add_diag_tensor, a=a), self.tree, _is_diag)
        return self.replace(tree=tree)

    def sum(self, axis=0, keepdims=None):
        # for vectors only for now
        assert self.treedef_l == _arr_treedef
        tree = jax.tree_map(partial(jnp.sum, axis=axis, keepdims=keepdims), self.tree)
        if keepdims:
            n_ax = 0
        else:
            n_ax = 1 if isinstance(axis, int) else len(axis)
        axes_l = self.axes_l - n_ax
        return self.replace(tree=tree, axes=(axes_l,) + self.axes[1:])

    def astype(self, dtype_tree):
        if isinstance(dtype_tree, PyTreeArrayT):
            dtype_tree = dtype_tree.tree
        tree = jax.tree_multimap(lambda x, y: x.astype(y), self.tree, dtype_tree)
        return self.replace(tree=tree)

    @property
    def dtype(self):
        return jax.tree_map(jnp.dtype, self.tree)

    # for the iterative solvers
    def __call__(self, vec):
        return self @ vec

    def _i_map(self, f, i):
        # treemap including and below i
        is_leaf = lambda x: jax.tree_structure(x) == self._leafdef(i)
        return jax.tree_map(f, self.tree, is_leaf=is_leaf)

    # TODO make the meaning of i consistent?
    def _i_amap(self, f, i):

        return self._i_map
        is_leaf = lambda x: jax.tree_structure(x) == self._leafdef(i)
        return jax.tree_map(f, self.tree, is_leaf=is_leaf)

    def swapaxes(self, i, j):

        tree = amap(partial(_swapaxes, i=i, j=j), self.tree, self.axes)

        tree_flat, _ = jax.tree_flatten(tree)
        new_treedefs = _swaptuple(self.treedefs, i, j)
        old_axes = self.axes

        new_axes = _swaptuple(old_axes, i, j)
        treedef = _treedefs_compose(*new_treedefs)
        tree = treedef.unflatten(tree_flat)

        return PyTreeArrayT(tree, new_treedefs, new_axes)


def _treedefs_compose(*treedefs):
    return reduce(lambda s1, s2: s1.compose(s2), treedefs)


def _swaptuple(t, i, j):
    if i == j:
        return t
    i, j = min(i, j), max(i, j)
    return t[:i] + t[j : j + 1] + t[i + 1 : j] + t[i : i + 1] + t[j + 1 :]


_arr_treedef = jax.tree_structure(jnp.zeros(0))  # TODO proper way to get * ??

# for a vector
def PyTreeArray(t):
    treedef_l = jax.tree_structure(t)
    # treedef_r = _arr_treedef
    axes_l = jax.tree_map(jnp.ndim, t)
    # axes_r = 0
    return PyTreeArrayT(t, (treedef_l,), (axes_l,))


# for the oks
def PyTreeArray2(t):
    treedef_l = _arr_treedef
    treedef_r = jax.tree_structure(t)
    axes_l = 1
    axes_r = jax.tree_map(lambda x: x - axes_l, jax.tree_map(jnp.ndim, t))
    return PyTreeArrayT(t, (treedef_l, treedef_r), (axes_l, axes_r))


def tree_allclose(t1, t2):
    return jax.tree_structure(t1) == jax.tree_structure(t2) and jax.tree_util.tree_reduce(lambda x, y: x and y, jax.tree_multimap(jnp.allclose, t1, t2))


# TODO eye_like / lazy add to diagonal
# TODO ignore flax FrozenDict in treedef comparison
# TODO ndim attr if treedefs are both * to emulate array


# def _valid_treedefs(pt):
#    ts = jax.tree_structure(pt.tree)
#    ts2 = reduce(lambda s1, s2: s1.compose(s2), pt.treedefs)
#    assert ts==ts2


def tree_multiply_outer(t1, t2):
    # np.multiply.outer
    return jax.tree_map(lambda l1: jax.tree_map(lambda l2: jnp.outer(l1, l2), t2), t1)


def size_multiply_outer(t1, t2):
    # np.multiply.outer
    return jax.tree_map(lambda l1: jax.tree_map(lambda l2: l1 * l2, t2), t1)


def build_size_tree(*shapedefs):
    return reduce(size_multiply_outer, shapedefs)


def multiply_outer(pt1, pt2):
    # np.multiply.outer
    tree = jax.tree_map(lambda t1: jax.tree_map(lambda t2: jnp.outer(t1, t2), pt2.tree), pt1.tree)
    treedefs = pt1.treedefs + pt2.treedefs
    axes = pt1.axes + pt2.axes
    return PyTreeArrayT(tree, treedefs, axes)


class leaf_tuple(tuple):
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


def _transpose(x, axes):
    # transpose a tensor with groups of axes whose size is given by axes
    l = _cumsum((0,) + axes[:-1])
    r = _cumsum(axes)
    i = tuple(map(lambda lr: tuple(range(*lr)), zip(l, r)))

    src = _flatten(i)
    dst = _flatten(i[::-1])
    trafo = tuple(map(lambda i: dst.index(i), src))

    return jnp.moveaxis(x, src, trafo)


def _cumsum(x):
    # cumsum of a tuple
    return reduce(lambda c, x_: c + (c[-1] + x_,) if c else (x_,), x, ())


def _flatten(t):
    # flatten a tuple of tuples
    return reduce(lambda x, y: x + y, t)
