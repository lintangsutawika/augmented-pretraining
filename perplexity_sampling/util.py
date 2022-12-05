import jax
import jax.numpy as jnp

from functools import partial

# 5x faster than NLTK's ngram
@partial(jax.jit, static_argnames=['n'])
def ngrams(sequence, n):

    assert len(sequence) >= n

    if type(sequence) == list:
        sequence = jnp.asarray(sequence)

    history = jax.lax.slice(sequence, [0], [n])
    sequence = jax.lax.slice(sequence, [n], [len(sequence)])

    def _f(init, x):
        y = jnp.append(init, x)
        return y[1:], y[1:]
    
    _, ys = jax.lax.scan(_f, history, sequence)

    return jnp.append(jnp.expand_dims(history, 0), ys, axis=0)