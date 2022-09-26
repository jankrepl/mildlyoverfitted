import numpy as np
import pytest


def get_arrays():
    """Create 4 arrays that are all similar but different.

    Returns
    -------
    a : np.ndarray
        Reference array.

    a_eps : np.ndarray
        Same shape as `a`, however, the values are slightly different.

    a_dim : np.ndarray
        One extra dimension compared to `a`, however, the values are the same.

    a_nan : np.ndarray
        Same shape and same values, however, one entry is set to `np.nan`.
    """
    eps = 1e-5

    a = np.array([[1.2, 5.12, 2.4], [5.5, 8.8, 1.55]])
    a_eps = a + eps
    a_dim = a[None, :]  # shape (1, 2, 3)
    a_nan = a.copy()
    a_nan[0, 1] = np.nan

    return a, a_eps, a_dim, a_nan


def test___eq__():
    a, *_ = get_arrays()

    with pytest.raises(ValueError):
        assert a == a


def test___eq__all():
    a, a_eps, a_dim, a_nan = get_arrays()

    assert (a == a).all()
    assert not (a == a_eps).all()
    assert (a == a_dim).all()
    assert not (a_nan == a_nan).all()


def test_array_equal():
    a, a_eps, a_dim, a_nan = get_arrays()

    assert np.array_equal(a, a)
    assert not np.array_equal(a, a_eps)
    assert not np.array_equal(a, a_dim)
    assert not np.array_equal(a_nan, a_nan)
    assert np.array_equal(a_nan, a_nan, equal_nan=True)


def test_allclose():
    a, a_eps, a_dim, a_nan = get_arrays()

    atol = 1e-5

    assert np.allclose(a, a, atol=atol)
    assert np.allclose(a, a_eps, atol=atol)
    assert np.allclose(a, a_dim, atol=atol)
    assert not np.allclose(a_nan, a_nan, atol=atol)
    assert np.allclose(a_nan, a_nan, atol=atol, equal_nan=True)


def test_testing_array_equal():
    a, a_eps, a_dim, a_nan = get_arrays()

    np.testing.assert_array_equal(a, a)
    # np.testing.assert_array_equal(a, a_eps)
    # np.testing.assert_array_equal(a, a_dim)
    np.testing.assert_array_equal(a_nan, a_nan)


def test_testing_allclose():
    a, a_eps, a_dim, a_nan = get_arrays()

    atol = 1e-5

    np.testing.assert_allclose(a, a, atol=atol)
    np.testing.assert_allclose(a, a_eps, atol=atol)
    # np.testing.assert_allclose(a, a_dim, atol=atol)
    np.testing.assert_allclose(a_nan, a_nan, atol=atol)
    # np.testing.assert_allclose(a_nan, a_nan, atol=atol, equal_nan=False)
