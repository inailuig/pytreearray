import pytest
import jax
import jax.numpy as jnp
import jax.flatten_util
import pytreearray as pta

from pytreearray._src.util import tree_allclose


@pytest.fixture
def vec():
    x = {"a": jnp.ones((3, 4)), "b": ((jnp.ones(2), jnp.ones((2, 2))))}
    px = pta.PyTreeArray1(x)
    return px


unary_funcs = {}
unary_funcs["neg"] = lambda x: -x
unary_funcs["add_l"] = lambda x: 1.23 + x
unary_funcs["add_r"] = lambda x: x + 1.23
unary_funcs["sub_l"] = lambda x: 1.23 - x
unary_funcs["sub_r"] = lambda x: x - 1.23
unary_funcs["mul_l"] = lambda x: 1.23 * x
unary_funcs["mul_r"] = lambda x: x * 1.23
unary_funcs["div_l"] = lambda x: 1.23 / x
unary_funcs["div_r"] = lambda x: x / 1.23
unary_funcs["pow"] = lambda x: x ** 1.23
unary_funcs["transpose"] = lambda x: x.transpose()
unary_funcs["conjugate"] = lambda x: x.conjugate()


@pytest.mark.parametrize("name, f", unary_funcs.items())
def test_unary(vec, name, f):
    vec_flat, unflatten = jax.flatten_util.ravel_pytree(vec.tree)
    actual = f(vec).tree
    expected = unflatten(f(vec_flat))

    assert tree_allclose(actual, expected)


binary_funcs = {}
binary_funcs["add"] = lambda x, y: x + y
binary_funcs["sub"] = lambda x, y: x - y
binary_funcs["mul"] = lambda x, y: x * y
binary_funcs["div"] = lambda x, y: x / y


@pytest.mark.parametrize("name, f", binary_funcs.items())
def test_binary(vec, name, f):
    vec_flat, unflatten = jax.flatten_util.ravel_pytree(vec.tree)
    actual = f(vec, vec).tree
    expected = unflatten(f(vec_flat, vec_flat))

    assert tree_allclose(actual, expected)
