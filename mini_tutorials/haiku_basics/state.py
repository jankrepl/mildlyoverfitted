from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


def foo(x: jnp.ndarray) -> jnp.ndarray:
    c = hk.get_parameter("c", x.shape, init=hk.initializers.RandomNormal(1))

    counter = hk.get_state(
        "counter", shape=[], dtype=jnp.int32, init=jnp.ones
    )
    hk.set_state("counter", counter + 1)
    res = c + x + counter

    return res 

foo_transformed = hk.transform_with_state(foo)
init_key = jax.random.PRNGKey(32)

x = jnp.ones((2, 5))
params, state = foo_transformed.init(init_key, x)

for i in range(2):
    print(f"After {i} iterations")

    res, state = foo_transformed.apply(params, state, None, x)
    print(state)
    print(res)

