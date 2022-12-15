import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from functools import partial

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


def init_matrix(num_vocab, k):

    seed_matrix = jnp.array([0])
    root_coordinate = jnp.array([[0]*k])
    matrix = BCOO((seed_matrix, root_coordinate), shape=tuple([num_vocab]*k))

    return matrix


@partial(jax.jit, static_argnames=['k'])
def increment_at_coordinate(matrix, ngram, k):

    if type(ngram) == list:
        ngram = jnp.asarray(ngram)

    assert k >= len(ngram)

    increment = jnp.array([1])

    if len(ngram) < k:
        indices = jnp.append(
            jnp.expand_dims(ngram, 0),
            jnp.array([[0]*(k-len(ngram))], dtype=jnp.int32),
            axis=1, dtype=jnp.int32
            )
    else:
        indices = jnp.expand_dims(ngram, 0)

    return BCOO((increment, indices), shape=matrix.shape)


@partial(jax.jit, static_argnames=['k'])
def get_value(matrix, ngram, k):

    if type(ngram) == list:
        ngram = jnp.asarray(ngram)

    assert k >= len(ngram)

    if len(ngram) < k:
        indices = jnp.append(
            jnp.expand_dims(ngram, 0),
            jnp.array([[0]*(k-len(ngram))], dtype=jnp.int32),
            axis=1,
            # dtype=jnp.int32
            )
    else:
        indices = jnp.expand_dims(ngram, 0)
    
    identity = jnp.array([1])

    x = BCOO((identity, indices), shape=matrix.shape)

    return jax.experimental.sparse.bcoo_multiply_sparse(matrix, x).sum()

def pad(sequence, n):

    if len(sequence) < n:
        if type(sequence) == list:
            sequence = jnp.asarray(sequence)

        indices = jnp.append(sequence, jnp.array([0]*(n-len(sequence)), dtype=jnp.int32))
        return indices
    else:
        return sequence


# @jax.jit
# @partial(jax.jit, static_argnames=['from_idx', 'to_idx'])
def get_slice(sequence, from_idx, to_idx):

    from_idx = from_idx * ~jnp.less(from_idx, 0)
    to_idx = to_idx * ~jnp.less(to_idx, 0)
    to_idx = to_idx

    return jax.lax.dynamic_slice(sequence, (from_idx,), (to_idx-from_idx+1,))
