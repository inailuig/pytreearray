import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util
import pytreearray as pta

from pytreearray._src.util import tree_allclose, tree_random_normal_like
import pytreearray.multiply

jax.config.update("jax_enable_x64", True)  # noqa: E402


@pytest.fixture
def vec():
    seed = 123
    key = jax.random.PRNGKey(seed)
    x = {
        "a": jnp.ones((3, 4), dtype=jnp.complex128),
        "b": ((jnp.ones((2), dtype=jnp.complex128), jnp.ones((2, 2), dtype=jnp.complex128))),
    }
    x = tree_random_normal_like(key, x)
    return pta.PyTreeArray1(x)


@pytest.fixture
def matr():
    seed = 123
    key = jax.random.PRNGKey(seed)
    A = {
        "a": jnp.ones((100, 3, 4), dtype=jnp.complex128),
        "b": ((jnp.ones((100, 2), dtype=jnp.complex128), jnp.ones((100, 2, 2), dtype=jnp.complex128))),
    }
    A = tree_random_normal_like(key, A)
    return pta.PyTreeArray2(A)


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
unary_funcs["transpose"] = lambda x: x.T
unary_funcs["conjugate"] = lambda x: x.conj()
unary_funcs["real"] = lambda x: x.real
unary_funcs["imag"] = lambda x: x.imag


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
binary_funcs["astype"] = lambda x, y: x.astype(y.dtype)


@pytest.mark.parametrize("name, f", binary_funcs.items())
def test_binary(vec, name, f):
    vec_flat, unflatten = jax.flatten_util.ravel_pytree(vec.tree)
    actual = f(vec, vec).tree
    expected = unflatten(f(vec_flat, vec_flat))

    assert tree_allclose(actual, expected)


def test_matmul(vec, matr):
    xp = vec
    Ap = matr
    x_flat, unflatten = jax.flatten_util.ravel_pytree(xp.tree)
    A_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(Ap.tree)

    actual = (Ap @ xp).tree
    expected = A_flat @ x_flat
    assert tree_allclose(actual, expected)


def test_matmul2(vec, matr):
    xp = vec
    Ap = matr
    x_flat, unflatten = jax.flatten_util.ravel_pytree(xp.tree)
    A_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(Ap.tree)

    actual = ((Ap.T @ Ap) @ xp).tree
    expected = unflatten(A_flat.T @ A_flat @ x_flat)
    assert tree_allclose(actual, expected)


def test_to_dense(matr):
    Ap = matr
    A_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(Ap.tree)

    actual = Ap.to_dense()
    expected = A_flat
    assert tree_allclose(actual, expected)


def test_add_diag(matr):
    Ap = matr
    A_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(Ap.tree)

    Ap = Ap.T @ Ap
    A_flat = A_flat.T @ A_flat
    s = 1.23
    actual = Ap.add_diag_scalar(s).to_dense()
    expected = A_flat + jnp.eye(A_flat.shape[0]) * s
    assert tree_allclose(actual, expected)


def test_sum(matr):
    Ap = matr
    A_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(Ap.tree)
    actual = Ap.sum(axis=0).to_dense()
    expected = A_flat.sum(axis=0)
    assert tree_allclose(actual, expected)


def test_multiply_outer(vec):
    x_flat, unflatten = jax.flatten_util.ravel_pytree(vec.tree)
    actual = pytreearray.multiply.outer(vec, vec).to_dense()
    expected = np.multiply.outer(x_flat, x_flat)
    assert tree_allclose(actual, expected)
