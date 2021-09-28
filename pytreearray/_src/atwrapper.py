from flax import struct
from numbers import Number

from .core import PyTreeArray, _arr_treedef
from .util import tree_allclose

import jax


@struct.dataclass
class _IndexUpdateHelper:
    pytreearr: PyTreeArray

    def __getitem__(self, indices):
        # only support indexing the first array for now
        # for multidimensional arrays in the treedefs[0] do
        # x[(1,1,1),]
        #
        # TODO multiple indices & not just arrays, e.g. x[(1,1), (2,3), 'a'], & partial indexing of those

        assert self.treedefs[0] == _arr_treedef
        # TODO support slice & ellipsis
        # TODO vectors with scalars?
        if isinstance(indices, int):
            indices = (indices,)
        assert self.axes[0] >= len(indices)
        return _IndexUpdateRef(self.pytreearr, indices)


@struct.dataclass
class _IndexUpdateRef:
    pytreearr: PyTreeArray
    index: "idx"

    def _index_full_dim(self):
        return self.pytreearr.axes[0] == len(self.indices)

    def _new_treedefs(self):
        if self._index_full_dim():
            return self.axes[1:]
        else:
            return (self.axes[0] - len(indices),) + self.axes[1:]

    def _new_axes(self):
        if self._index_full_dim():
            return self.treedefs[1:]
        else:
            return self.treedefs

    def _check(self, val):
        if isinstance(val, PyTreeArray):

            def _equal(xy):
                x, y = xy
                return x == y

            assert all(map(_equal, zip(self._new_treedefs(), val.treedefs)))
            assert tree_allclose(val.axes, self._new_axes())
        else:
            if len(self._new_treedefs()) == 0:
                raise NotImplementedError
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
            return jax.tree_multimap(_update, self.pytreearr, _val)

    def get(self):
        def _get(x):
            return x.at[self.index]

        tree = jax.tree_map(_get, self.pytreearr.tree)
        if self.pytreearr.axes[0] == len(self.indices):
            return PyTreeArray(tree, self.treedefs[1:], self.axes[1:])
        else:
            return PyTreeArray(tree, self.treedefs, (self.axes[0] - len(indices),) + self.axes[1:])
