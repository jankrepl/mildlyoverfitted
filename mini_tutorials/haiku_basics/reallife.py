from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


def foo(x: jnp.ndarray) -> jnp.ndarray:
    mlp = hk.nets.MLP([4, 5, 1])

    loss = mlp(x).mean()

    return loss


foo_transformed = hk.without_apply_rng(hk.transform(foo))

init_key = jax.random.PRNGKey(3452)
x = jnp.ones((2, 3))
params = foo_transformed.init(init_key, x)

grad_foo = jax.jit(jax.grad(foo_transformed.apply))

grads = grad_foo(params, x)
