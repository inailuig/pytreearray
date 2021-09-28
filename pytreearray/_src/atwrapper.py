from flax import struct
from numbers import Number

from .core import PyTreeArray, _arr_treedef
from .util import tree_allclose

import jax
import jax.numpy as jnp


@struct.dataclass
class _IndexUpdateHelper:
    pytreearr: PyTreeArray

    def __getitem__(self, indices):
        # only support indexing the first dim for now
        # TODO can we make it work so that multiple indices use .get[].get[]
        # by adding .at to _IndexUpdateRef ?
        # What already works is multiple .at[].get().at[].get(), but not for the setter :/

        assert self.pytreearr.treedefs[0] == _arr_treedef
        if not isinstance(indices, tuple):
            indices = (indices,)
        # TODO return scalars as scalars? (now they are PyTreeArrays of a scalar)

        assert self.pytreearr.axes[0] >= len(indices)
        return _IndexUpdateRef(self.pytreearr, indices)


@struct.dataclass
class _IndexUpdateRef:
    pytreearr: PyTreeArray
    index: "idx"

    def _index_full_dim(self):
        return self.pytreearr.axes[0] == len(self.index)

    def _new_axes(self):
        if self._index_full_dim():
            return self.pytreearr.axes[1:]
        else:
            return (self.pytreearr.axes[0] - len(self.index),) + self.pytreearr.axes[1:]

    def _new_treedefs(self):
        if self._index_full_dim():
            return self.pytreearr.treedefs[1:]
        else:
            return self.pytreearr.treedefs

    def _check(self, val):
        if isinstance(val, PyTreeArray):

            def _equal(xy):
                x, y = xy
                return x == y

            assert all(map(_equal, zip(self._new_treedefs(), val.treedefs)))
            assert tree_allclose(val.axes, self._new_axes())
        else:
            if len(self._new_treedefs()) == 0:
                pass
            elif len(self._new_treedefs()) == 1:
                assert jax.tree_structure(val) == self._new_treedefs()[0]
                assert tree_allclose(jax.tree_map(jnp.ndim, val), self._new_axes()[0])

            else:
                raise NotImplementedError

    def set(self, val):

        self._check(val)

        if isinstance(val, Number) or hasattr(val, "shape") and val.ndim == 0:

            def _update(x):
                return x.at[self.index].set(val)

            return jax.tree_map(_update, self.pytreearr)

        else:

            def _update(x, val):
                return x.at[self.index].set(val)

            _val = val if not isinstance(val, PyTreeArray) else val.tree
            _tree = jax.tree_multimap(_update, self.pytreearr.tree, _val)
            return self.pytreearr.replace(tree=_tree)

    def get(self):
        def _get(x):
            return x.at[self.index].get()

        _tree = jax.tree_map(_get, self.pytreearr.tree)
        return PyTreeArray(_tree, self._new_treedefs(), self._new_axes())
