from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


def foo(x: jnp.ndarray) -> jnp.ndarray:
    c = hk.get_parameter("c", x.shape, init=hk.initializers.RandomNormal(1))

    res = c + x

    key = hk.next_rng_key()
    mask = jax.random.bernoulli(key, 0.5, x.shape)

    return res * mask * 2


foo_transformed = hk.transform(foo)

init_key = jax.random.PRNGKey(24)
apply_key_seq = hk.PRNGSequence(init_key)

x = jnp.ones((2, 5))
params = foo_transformed.init(init_key, x)

for _ in range(2):
    res = foo_transformed.apply(params, next(apply_key_seq), x)
    print(res)
